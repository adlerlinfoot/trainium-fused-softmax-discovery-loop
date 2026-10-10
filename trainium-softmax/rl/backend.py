"""Trainable HuggingFace policy, replacing Tinker sampling/training calls."""
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch


def importance_loss(current_logprobs, sampled_logprobs, advantages, loss_fn='importance_sampling'):
    # Tinker's documented token-wise loss: SUM, not mean over completion tokens.
    ratio = torch.exp(current_logprobs - sampled_logprobs)
    if loss_fn == 'importance_sampling': return -(ratio * advantages).sum()
    if loss_fn == 'ppo':
        return -torch.minimum(ratio * advantages, ratio.clamp(.8,1.2) * advantages).sum()
    raise ValueError(loss_fn)


def static_experts_forward(self, hidden_states, router_indices=None, routing_weights=None):
    """Algebraically identical dense routing; fixed shapes for XLA.

    Evaluates every expert on all tokens and masks outputs. Avoids the stock
    forward's data-dependent nonzero/where Python loop; trades compute for a
    simple training compatibility path. Expert weights and routing stay intact.
    """
    original_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1,self.hidden_size)
    output = torch.zeros_like(hidden_states)
    for expert in range(self.num_experts):
        global_expert = expert + getattr(self, 'expert_start', 0)
        weight = routing_weights[:,global_expert:global_expert+1] if routing_weights.shape[-1] == getattr(self, 'global_num_experts', self.num_experts) else ((router_indices == global_expert).to(routing_weights.dtype) * routing_weights).sum(-1,keepdim=True)
        gate_up = hidden_states @ self.gate_up_proj[expert] + self.gate_up_proj_bias[expert]
        # 4.57 GptOss uses the same gate definition, but lacks _apply_gate.
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit,max=self.limit)
        gated = (up + 1) * gate * torch.sigmoid(gate * self.alpha)
        y = gated @ self.down_proj[expert] + self.down_proj_bias[expert]
        output = output + y * weight
    return output.reshape(original_shape)



class CopyToExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xm):
        ctx.xm=xm
        return x
    @staticmethod
    def backward(ctx, grad):
        return ctx.xm.all_reduce(ctx.xm.REDUCE_SUM,grad),None

class ReduceFromExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,xm):
        return xm.all_reduce(xm.REDUCE_SUM,x)
    @staticmethod
    def backward(ctx,grad):
        return grad,None


def shard_experts(model,rank,world,xm):
    """Shard frozen expert parameters before moving them to accelerator memory.

    Attention and LoRA parameters are replicated. Expert outputs are summed,
    and their input gradients are summed, preserving the unsharded function.
    """
    import types
    def parallel_mlp(self,hidden_states):
        hidden_states=CopyToExperts.apply(hidden_states,xm)
        router_scores,router_indices=self.router(hidden_states)
        local=self.experts(hidden_states,router_indices=router_indices,routing_weights=router_scores)
        return ReduceFromExperts.apply(local,xm),router_scores
    for mod in model.modules():
        if mod.__class__.__name__=='GptOssExperts':
            total=mod.num_experts
            if total%world: raise ValueError('Expert count must divide parallel world size')
            count=total//world; start=rank*count
            for name in ('gate_up_proj','gate_up_proj_bias','down_proj','down_proj_bias'):
                value=getattr(mod,name)
                setattr(mod,name,torch.nn.Parameter(value.detach()[start:start+count].clone(),requires_grad=False))
            mod.global_num_experts=total; mod.num_experts=count; mod.expert_start=start
        if mod.__class__.__name__=='GptOssMLP': mod.forward=types.MethodType(parallel_mlp,mod)

class Policy:
    def __init__(self, model_path, device='xla', rank=32, seed=0, static_experts=False,
                 tiny=False, max_context=2048, parallel=False, cached_sampling=False, checkpoint_activations=False,
                 tensor_parallel_size=1,tiny_kv_heads=None,partial=False):
        self.version = 0
        self.seed = seed
        self.max_context = max_context
        self.device_kind = device
        self.model_path = str(model_path)
        self.rank = rank
        self.static_experts = static_experts
        parallel = parallel or tensor_parallel_size > 1
        self.parallel = parallel
        self.tensor_parallel_size = tensor_parallel_size
        self.cached_sampling = cached_sampling
        self.checkpoint_activations = checkpoint_activations
        self.world_size = 1
        self.rank_id = 0
        self.partial = partial
        self.xm = None
        random.seed(seed); torch.manual_seed(seed)
        if device == 'xla':
            os.environ.setdefault('PJRT_DEVICE','NEURON')
            if parallel:
                local_rank=int(os.environ.get('LOCAL_RANK','0'))
                first=int(os.environ.get('TRAINIUM_FIRST_CORE','0' if tensor_parallel_size>1 else '2'))
                os.environ['NEURON_RT_VISIBLE_CORES']=str(first+local_rank)
            import torch_neuronx  # registers the Neuron PJRT plugin
            import torch_xla.core.xla_model as xm
            self.xm = xm
            if parallel:
                import torch.distributed as dist
                dist.init_process_group('xla')
                self.world_size = dist.get_world_size()
                self.rank_id = dist.get_rank()
                if tensor_parallel_size>1 and self.world_size!=tensor_parallel_size:
                    raise ValueError('Distributed world does not match TP degree')
            self.device = xm.xla_device()
        elif device == 'cpu':
            self.device = torch.device('cpu')
            if parallel:
                import torch.distributed as dist
                if not dist.is_initialized(): raise ValueError('CPU parallel tests require an initialized process group')
                self.world_size=dist.get_world_size(); self.rank_id=dist.get_rank()
                # Test adapter only; production uses the Neuron collectives.
                from tensor_parallel import CpuCollectives
                self.xm=CpuCollectives()
        else: raise ValueError('Only explicit xla or cpu supported')
        if tensor_parallel_size>1 and self.world_size!=tensor_parallel_size:
            raise ValueError('Distributed world does not match TP degree')
        from transformers import AutoModelForCausalLM, AutoTokenizer, GptOssConfig, Mxfp4Config
        from peft import LoraConfig, get_peft_model
        if tiny:
            config = GptOssConfig(hidden_size=32,intermediate_size=32,num_hidden_layers=1,
                num_attention_heads=4,num_key_value_heads=tiny_kv_heads or (4 if tensor_parallel_size>2 else 2),head_dim=8,num_local_experts=4,
                num_experts_per_tok=2,vocab_size=128,max_position_embeddings=256)
            config._attn_implementation = 'eager'
            model = AutoModelForCausalLM.from_config(config)
            self.tokenizer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_class = AutoModelForCausalLM
            if partial:
                from partial_model import PartialGptOssForCausalLM
                model_class = PartialGptOssForCausalLM
            loaded = model_class.from_pretrained(model_path,
                torch_dtype=torch.bfloat16,quantization_config=Mxfp4Config(dequantize=True),
                attn_implementation='eager',device_map='cpu',output_loading_info=partial)
            if partial:
                model, loading = loaded
                if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
                    raise RuntimeError(f'Partial checkpoint did not load exactly: {loading}')
            else:
                model = loaded
        self.tiny = tiny
        self.selected_layers = list(model.config.verification_layer_indices) if partial else None
        if parallel and tensor_parallel_size == 1:
            if not static_experts: raise ValueError('Expert parallelism requires static experts')
            shard_experts(model,self.rank_id,self.world_size,self.xm)
        if static_experts:
            import types
            for mod in model.modules():
                if mod.__class__.__name__ == 'GptOssExperts':
                    mod.forward = types.MethodType(static_experts_forward,mod)
        # Explicit adaptation scope. Tinker doesn't expose its exact target list.
        self.model = get_peft_model(model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=0.,
            target_modules=['q_proj','k_proj','v_proj','o_proj'],task_type='CAUSAL_LM'))
        if tensor_parallel_size > 1:
            if not static_experts: raise ValueError('TP requires fixed-shape experts')
            from tensor_parallel import shard_tensors
            shard_tensors(self.model,self.rank_id,self.world_size,self.xm)
        self.model.config.use_cache = False
        if checkpoint_activations:
            if device=='xla':
                from functools import partial
                from torch_xla.utils.checkpoint import checkpoint
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
                self.model.enable_input_require_grads()
                # Torch's generic checkpoint assumes torch.xla exists; this
                # SDK provides the XLA-aware reentrant implementation instead.
                for module in self.model.modules():
                    if hasattr(module,'_gradient_checkpointing_func'):
                        module._gradient_checkpointing_func=partial(checkpoint,use_reentrant=True,preserve_rng_state=False)
            else:
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        self.model.to(self.device)
        self.sync()
        self.optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
            lr=4e-5,betas=(.9,.95),eps=1e-8,weight_decay=0.)

    def sync(self):
        if self.xm is not None:
            self.xm.mark_step()
            self.xm.wait_device_ops()

    def encode(self, prompt):
        if self.tiny: raise RuntimeError('Tiny test model has no production tokenizer')
        from vendor.renderers import GptOssRenderer
        renderer=GptOssRenderer(self.tokenizer,use_system_prompt=True,reasoning_effort='high',current_date='2026-10-10')
        model_input=renderer.build_generation_prompt([{'role':'user','content':prompt}])
        return [token for chunk in model_input.chunks for token in chunk.tokens]

    async def sample_async(self,prompt,num_samples,sampling_params):
        from types import SimpleNamespace
        if num_samples != 1 or sampling_params.temperature != 1.:
            raise ValueError('Adapter supports reference temperature=1 and one sample per call')
        tokens=[t for chunk in prompt.chunks for t in chunk.tokens]
        result=self.sample(tokens,sampling_params.max_tokens,stop=sampling_params.stop)
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=result['tokens'],logprobs=result['logprobs'])])

    def sample_two_phase(self,prompt_tokens,max_tokens):
        import asyncio
        from vendor.completers import TwoPhaseTokenCompleter
        from vendor.token_api import ModelInput,EncodedTextChunk
        completer=TwoPhaseTokenCompleter(self,self.tokenizer,len(prompt_tokens)+max_tokens,
            temperature=1.,context_window=self.max_context,context_buffer=50)
        from vendor.renderers import GptOssRenderer
        stop=GptOssRenderer(self.tokenizer).get_stop_sequences()
        action=asyncio.run(completer(ModelInput([EncodedTextChunk(prompt_tokens)]),stop))
        return dict(prompt_tokens=prompt_tokens,tokens=action.tokens,logprobs=action.logprobs,
            mask=action.mask,policy_version=self.version,adapter_sha256=self.fingerprint())

    def fingerprint(self):
        self.sync()
        h = hashlib.sha256()
        for n,p in self.model.named_parameters():
            if p.requires_grad:
                h.update(n.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
        if self.tensor_parallel_size>1:
            digest=torch.tensor(list(h.digest()),dtype=torch.int64,device=self.device)
            gathered=self.xm.all_gather(digest).cpu().tolist()
            return hashlib.sha256(bytes(gathered)).hexdigest()
        return h.hexdigest()

    def logprobs(self, prompt_tokens, tokens, base=False):
        seq = prompt_tokens + tokens
        if len(seq) > self.max_context: raise ValueError('Sequence exceeds fixed context limit')
        ids = torch.tensor([seq[:-1]],dtype=torch.long,device=self.device)
        labels = torch.tensor([seq[1:]],dtype=torch.long,device=self.device)
        if base:
            with self.model.disable_adapter(): logits = self.model(input_ids=ids,use_cache=False).logits
        else: logits = self.model(input_ids=ids,use_cache=False).logits
        logp = torch.log_softmax(logits.float(),dim=-1).gather(-1,labels.unsqueeze(-1)).squeeze(-1)
        return logp[0,len(prompt_tokens)-1:]

    def sample(self,prompt_tokens,max_tokens=512,stop=None):
        # Same in-memory model for sampling and updates; no stale serving weights.
        self.model.eval()
        # Hash before constructing the decode graph. mark_step inside a hash
        # taken while unused model outputs are live compiles those outputs too.
        adapter_sha256=self.fingerprint()
        tokens=[]; logprobs=[]
        stop_patterns=[[s] if isinstance(s,int) else self.tokenizer.encode(s,add_special_tokens=False)
                       for s in (self.eos_ids() if stop is None else stop)]
        if len(prompt_tokens)+max_tokens > self.max_context: raise ValueError('Prompt+output exceeds context limit')
        cache=None
        if self.cached_sampling:
            from transformers import StaticCache
            cache=StaticCache(config=self.model.config,max_cache_len=self.max_context)
        with torch.no_grad():
            for _ in range(max_tokens):
                if cache is None:
                    ids = torch.tensor([prompt_tokens+tokens],dtype=torch.long,device=self.device)
                    outputs=self.model(input_ids=ids,use_cache=False)
                else:
                    ids=torch.tensor([prompt_tokens if not tokens else [tokens[-1]]],dtype=torch.long,device=self.device)
                    positions=torch.arange(len(prompt_tokens),device=self.device) if not tokens else torch.tensor([len(prompt_tokens)+len(tokens)-1],device=self.device)
                    outputs=self.model(input_ids=ids,past_key_values=cache,cache_position=positions,use_cache=True)
                    # Materialize persistent KV tensors before the next decode;
                    # otherwise its lazy graph can replay the whole prefill.
                    self.sync()
                logits=outputs.logits[0,-1].float()
                # CPU sampling is explicit; model forward/logits execute on requested device.
                # Temperature exactly 1, no truncation, as reference default requires.
                lp = torch.log_softmax(logits,dim=-1).cpu()
                token = int(torch.multinomial(lp.exp(),1).item())
                tokens.append(token); logprobs.append(float(lp[token]))
                if any(pattern and tokens[-len(pattern):]==pattern for pattern in stop_patterns): break
        return dict(prompt_tokens=prompt_tokens,tokens=tokens,logprobs=logprobs,
                    policy_version=self.version,adapter_sha256=adapter_sha256)

    def eos_ids(self):
        value = self.model.generation_config.eos_token_id
        if value is None: return []
        return value if isinstance(value,list) else [value]

    def update(self,rollouts,advantages,lr=4e-5,kl_coef=.1,loss_fn='importance_sampling'):
        if len(rollouts)!=len(advantages): raise ValueError('Rollout/advantage count mismatch')
        if any(r['policy_version']!=self.version for r in rollouts):
            raise ValueError('Stale rollout policy version')
        before=self.fingerprint()
        if any(r.get('adapter_sha256') != before for r in rollouts):
            raise ValueError('Rollout adapter fingerprint does not match learner checkpoint')
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        for group in self.optimizer.param_groups: group['lr']=lr
        # Match reference KL advantage adjustment across all sampled completion tokens.
        diffs=[]
        if kl_coef:
            with torch.no_grad():
                for r in rollouts:
                    base=self.logprobs(r['prompt_tokens'],r['tokens'],base=True).cpu()
                    mask=torch.tensor(r.get('mask',[1.]*len(r['tokens'])))
                    diffs.append((torch.tensor(r['logprobs'])-base)*mask)
            mean_diff = sum(float(d.sum()) for d in diffs)/sum(sum(r.get('mask',[1.]*len(r['tokens']))) for r in rollouts)
        else: mean_diff=0.
        losses=[]
        for i,(r,adv) in enumerate(zip(rollouts,advantages)):
            if not r['tokens']: continue
            current=self.logprobs(r['prompt_tokens'],r['tokens'])
            old=torch.tensor(r['logprobs'],dtype=torch.float32,device=self.device)
            mask=torch.tensor(r.get('mask',[1.]*len(r['tokens'])),device=self.device)
            token_adv=torch.full_like(old,float(adv))*mask
            if kl_coef:
                token_adv=token_adv+kl_coef*mask*(mean_diff-diffs[i].to(self.device))
            loss=importance_loss(current,old,token_adv,loss_fn)
            loss.backward(); self.sync()
            losses.append(float(loss.detach().cpu()))
        if self.tensor_parallel_size>1:
            from tensor_parallel import reduce_replicated_gradients,is_replicated_adapter
            reduce_replicated_gradients(self.model,self.xm)
            self.sync()
        grad_sq=0.; finite=True
        for name,p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g=p.grad.detach().float().cpu()
                finite=finite and bool(torch.isfinite(g).all())
                contribution=float(g.square().sum())
                if self.tensor_parallel_size>1 and is_replicated_adapter(name): contribution/=self.world_size
                grad_sq+=contribution
        if self.tensor_parallel_size>1:
            totals=self.xm.all_reduce(self.xm.REDUCE_SUM,torch.tensor([grad_sq,float(not finite)],dtype=torch.float32,device=self.device)).cpu()
            grad_sq=float(totals[0]); finite=float(totals[1])==0
        if not finite: raise RuntimeError('Nonfinite adapter gradients; optimizer not applied')
        if grad_sq == 0: raise RuntimeError('Zero adapter gradients; optimizer not applied')
        self.optimizer.step(); self.sync()
        self.version+=1
        after=self.fingerprint()
        fallback_ops=[]
        if self.device_kind=='xla':
            from torch_xla.debug.metrics import executed_fallback_ops
            fallback_ops=executed_fallback_ops()
        return dict(loss_sum=sum(losses),gradient_norm=grad_sq**.5,gradients_finite=finite,
                    parameters_changed=before!=after,before_sha256=before,after_sha256=after,
                    policy_version=self.version,kl_policy_base=mean_diff,fallback_ops=fallback_ops,
                    device=str(self.device),world_size=self.world_size,tensor_parallel_size=self.tensor_parallel_size,
                    trainable_parameters_local=sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def save(self,directory):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
        self.sync()
        self.model.save_pretrained(directory/'adapter')
        # Move optimizer tensors to CPU for portability.
        def cpu(value):
            if torch.is_tensor(value): return value.detach().cpu()
            if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
            if isinstance(value,list): return [cpu(v) for v in value]
            return value
        torch.save({'optimizer':cpu(self.optimizer.state_dict()),'rng':torch.get_rng_state(),
            'python_rng':random.getstate(),'version':self.version},directory/'training.pt')
        (directory/'policy.json').write_text(json.dumps(dict(model_path=self.model_path,rank=self.rank,
            static_experts=self.static_experts,tiny=self.tiny,max_context=self.max_context,cached_sampling=self.cached_sampling,checkpoint_activations=self.checkpoint_activations,parallel=self.parallel,world_size=self.world_size,tensor_parallel_size=self.tensor_parallel_size,
            version=self.version,adapter_sha256=self.fingerprint(),partial=self.partial,
            selected_layers=self.selected_layers),indent=2))

    def restore(self,directory):
        directory=Path(directory)
        saved_policy=json.loads((directory/'policy.json').read_text())
        if saved_policy.get('partial',False)!=self.partial or saved_policy.get('selected_layers')!=self.selected_layers:
            raise RuntimeError('Checkpoint verification scope or selected layers mismatch')
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        state=load_file(str(directory/'adapter'/'adapter_model.safetensors'))
        set_peft_model_state_dict(self.model,state)
        training=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
        self.optimizer.load_state_dict(training['optimizer'])
        torch.set_rng_state(training['rng']); random.setstate(training['python_rng'])
        self.version=training['version']; self.sync()
        expected=json.loads((directory/'policy.json').read_text())['adapter_sha256']
        if self.fingerprint()!=expected: raise RuntimeError('Restored adapter hash mismatch')
