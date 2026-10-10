"""Real-checkpoint backend gate. --tiny is explicitly a separate compatibility test."""
import argparse
import json
import traceback
from pathlib import Path
import time
from backend import Policy

def main(a):
    import os
    if a.parallel or a.tensor_parallel_size>1: a.out=a.out/f'rank-{os.getenv("RANK","0")}'
    a.out.mkdir(parents=True,exist_ok=False)
    events=[]
    def emit(phase,**data):
        record=dict(phase=phase,time=time.time(),**data); events.append(record)
        print(json.dumps(record),flush=True)
        (a.out/'events.json').write_text(json.dumps(events,indent=2))
    try:
        emit('initialize',model=a.model,device=a.device,tiny=a.tiny,static_experts=a.static_experts,
             tensor_parallel_size=a.tensor_parallel_size,cached_sampling=a.cached_sampling,checkpoint_activations=a.checkpoint_activations,
             scope='partial-model verification' if a.partial else 'existing probe')
        policy=Policy(a.model,a.device,rank=32,static_experts=a.static_experts,tiny=a.tiny,max_context=128,parallel=a.parallel,
            cached_sampling=a.cached_sampling,checkpoint_activations=a.checkpoint_activations,tensor_parallel_size=a.tensor_parallel_size,partial=a.partial)
        attention_settings = [dict(original_index=layer.self_attn.layer_idx,
            attention_type=layer.attention_type,sliding_window=layer.self_attn.sliding_window)
            for layer in policy.model.base_model.model.model.layers] if a.partial else None
        emit('loaded',adapter_sha256=policy.fingerprint(),selected_layers=policy.selected_layers,
            attention_settings=attention_settings)
        prompt=[1,2,3] if a.tiny else policy.tokenizer.encode('Write an NKI softmax kernel.',add_special_tokens=False)[:8]
        rollout=policy.sample(prompt,max_tokens=a.sample_tokens)
        emit('sampled',**rollout)
        # Synthetic +1 advantage isolates gradient/update mechanics, not task RL success.
        metrics=policy.update([rollout],[1.],kl_coef=0.)
        emit('updated',**metrics)
        if not metrics['parameters_changed'] or metrics['gradient_norm']<=0:
            raise RuntimeError('No verified nonzero gradient and parameter change')
        policy.save(a.out/'checkpoint')
        post=policy.sample(prompt,max_tokens=a.sample_tokens)
        assert post['policy_version']==1 and post['adapter_sha256']==metrics['after_sha256']
        emit('sampled_updated_policy',**post)
        policy.restore(a.out/'checkpoint')
        emit('checkpoint_restored',adapter_sha256=policy.fingerprint())
        result=dict(status='passed',real_checkpoint=not a.tiny,model=a.model,
            scope='partial-model verification' if a.partial else 'existing probe',selected_layers=policy.selected_layers,
            attention_settings=attention_settings,
            on_trainium=a.device=='xla',**metrics)
    except Exception as exc:
        result=dict(status='failed',real_checkpoint=not a.tiny,model=a.model,device=a.device,
            scope='partial-model verification' if a.partial else 'existing probe',
            error=str(exc),traceback=traceback.format_exc())
        emit('failed',error=str(exc))
    (a.out/'result.json').write_text(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='openai/gpt-oss-20b')
    p.add_argument('--device',choices=['cpu','xla'],default='xla')
    p.add_argument('--partial',action='store_true')
    p.add_argument('--out',type=Path,required=True); p.add_argument('--tiny',action='store_true')
    p.add_argument('--static-experts',action='store_true'); p.add_argument('--parallel',action='store_true')
    p.add_argument('--cached-sampling',action='store_true'); p.add_argument('--checkpoint-activations',action='store_true')
    p.add_argument('--sample-tokens',type=int,default=1)
    p.add_argument('--tensor-parallel-size',type=int,choices=[1,2,4],default=1)
    raise SystemExit(main(p.parse_args()))
