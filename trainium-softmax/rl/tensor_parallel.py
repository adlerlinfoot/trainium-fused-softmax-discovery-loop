"""GPT-OSS tensor sharding using Neuron/XLA autograd collectives.

Attention heads and every expert's intermediate channels are partitioned.
Embedding, LM head, normalization and router weights remain replicated.
LoRA keeps one global rank: column A and row B are replicated; their gradients
must be summed once after accumulating the complete rollout batch.
"""
import types
import torch


class CpuCollectives:
    """CPU distributed parity tests only; never selected by the XLA backend."""
    REDUCE_SUM='sum'
    @staticmethod
    def all_reduce(op,value):
        import torch.distributed as dist
        result=value.clone()
        dist.all_reduce(result)
        return result
    @staticmethod
    def all_gather(value):
        import torch.distributed as dist
        parts=[torch.empty_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(parts,value)
        return torch.cat(parts)
    @staticmethod
    def mark_step(): pass
    @staticmethod
    def wait_device_ops(): pass


def shard_tensors(model, rank, world, xm):
    from backend import CopyToExperts, ReduceFromExperts

    def frozen(value):
        return torch.nn.Parameter(value.detach().clone(), requires_grad=False)

    def slice_linear(layer, column):
        base = layer.base_layer
        axis = 0 if column else 1
        size = base.weight.shape[axis]
        if size % world:
            raise ValueError('Linear dimension does not divide TP degree')
        width = size // world
        sl = [slice(None), slice(None)]
        sl[axis] = slice(rank * width, (rank + 1) * width)
        base.weight = frozen(base.weight[tuple(sl)])
        if column:
            base.out_features = layer.out_features = width
            if base.bias is not None:
                base.bias = frozen(base.bias[rank * width:(rank + 1) * width])
            for adapter in layer.lora_B:
                b = layer.lora_B[adapter]
                b.weight = torch.nn.Parameter(b.weight.detach()[rank * width:(rank + 1) * width].clone())
                b.out_features = width
                layer.lora_A[adapter].weight.tp_replicated = True
        else:
            base.in_features = layer.in_features = width
            if base.bias is not None:
                base.bias = frozen(base.bias / world)
            for adapter in layer.lora_A:
                a = layer.lora_A[adapter]
                a.weight = torch.nn.Parameter(a.weight.detach()[:, rank * width:(rank + 1) * width].clone())
                a.in_features = width
                layer.lora_B[adapter].weight.tp_replicated = True

    def attention_forward(self, hidden_states, *args, **kwargs):
        hidden_states = CopyToExperts.apply(hidden_states, xm)
        local, weights = self.tp_original_forward(hidden_states, *args, **kwargs)
        return ReduceFromExperts.apply(local, xm), weights

    def mlp_forward(self, hidden_states):
        hidden_states = CopyToExperts.apply(hidden_states, xm)
        scores, indices = self.router(hidden_states)
        local = self.experts(hidden_states, router_indices=indices, routing_weights=scores)
        return ReduceFromExperts.apply(local, xm), scores

    for module in model.modules():
        if module.__class__.__name__ == 'GptOssAttention':
            heads = module.sinks.numel()
            kv_heads = module.k_proj.base_layer.out_features // module.head_dim
            if heads % world or kv_heads % world:
                raise ValueError('Query and KV heads must divide TP degree')
            for name in ('q_proj', 'k_proj', 'v_proj'):
                slice_linear(getattr(module, name), True)
            slice_linear(module.o_proj, False)
            count = heads // world
            module.sinks = frozen(module.sinks[rank * count:(rank + 1) * count])
            module.tp_original_forward = module.forward
            module.forward = types.MethodType(attention_forward, module)
        elif module.__class__.__name__ == 'GptOssExperts':
            size = module.expert_dim
            if size % world:
                raise ValueError('Expert intermediate width must divide TP degree')
            width = size // world
            start, end = rank * width, (rank + 1) * width
            module.gate_up_proj = frozen(module.gate_up_proj[..., 2 * start:2 * end])
            module.gate_up_proj_bias = frozen(module.gate_up_proj_bias[..., 2 * start:2 * end])
            module.down_proj = frozen(module.down_proj[:, start:end, :])
            module.down_proj_bias = frozen(module.down_proj_bias / world)
            module.expert_dim = module.intermediate_size = width
        elif module.__class__.__name__ == 'GptOssMLP':
            module.forward = types.MethodType(mlp_forward, module)


def reduce_replicated_gradients(model, xm):
    for name, parameter in model.named_parameters():
        if is_replicated_adapter(name) and parameter.grad is not None:
            parameter.grad = xm.all_reduce(xm.REDUCE_SUM, parameter.grad)


def is_replicated_adapter(name):
    return any(f'{projection}.lora_A.' in name for projection in ('q_proj','k_proj','v_proj')) or 'o_proj.lora_B.' in name
