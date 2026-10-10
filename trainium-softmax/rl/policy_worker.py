"""Separate sampling/training phases release the chip before kernel evaluation."""
import argparse
import json
from pathlib import Path
import traceback
import time
from backend import Policy


def main(a):
    import os
    if a.parallel or a.tensor_parallel_size>1: a.out=a.out/f'rank-{os.getenv("RANK","0")}'
    a.out.mkdir(parents=True,exist_ok=False)
    config=json.loads(a.request.read_text())
    try:
        p=Policy(config['model_path'],config.get('device','xla'),rank=config.get('rank',32),
            static_experts=config.get('static_experts',True),max_context=config.get('max_context',2048),
            seed=config.get('seed',0),parallel=a.parallel,cached_sampling=True,checkpoint_activations=True,tensor_parallel_size=a.tensor_parallel_size)
        if config.get('checkpoint'):
            checkpoint=Path(config['checkpoint'])
            if p.parallel: checkpoint=checkpoint/f'rank-{p.rank_id}'/'checkpoint'
            p.restore(checkpoint)
        if a.mode=='sample':
            groups=[]
            for group_index,text in enumerate(config['prompts']):
                encoded=p.encode(text)
                group=[]
                for rollout_index in range(config['group_size']):
                    print(json.dumps(dict(phase='rollout_started',group=group_index,
                        rollout=rollout_index,time=time.time())),flush=True)
                    r=p.sample_two_phase(encoded,config['max_tokens'])
                    r['text']=p.tokenizer.decode(r['tokens'],skip_special_tokens=False)
                    group.append(r)
                    print(json.dumps(dict(phase='rollout_completed',group=group_index,
                        rollout=rollout_index,tokens=len(r['tokens']),time=time.time())),flush=True)
                groups.append(group)
            # Capture unchanged rollout weights and RNG advanced by sampling.
            p.save(a.out/'checkpoint')
            result=dict(status='passed',groups=groups,policy_version=p.version,
                checkpoint=str(a.out/'checkpoint'),adapter_sha256=p.fingerprint())
        else:
            metrics=p.update(config['rollouts'],config['advantages'],lr=config.get('learning_rate',4e-5),
                kl_coef=config.get('kl_coef',.1),loss_fn=config.get('loss_fn','importance_sampling'))
            p.save(a.out/'checkpoint')
            result=dict(status='passed',checkpoint=str(a.out/'checkpoint'),**metrics)
    except Exception as exc:
        result=dict(status='failed',error=str(exc),traceback=traceback.format_exc())
    (a.out/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','traceback')}),flush=True)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['sample','train'])
    parser.add_argument('--parallel',action='store_true'); parser.add_argument('--request',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--tensor-parallel-size',type=int,choices=[1,2,4],default=1)
    raise SystemExit(main(parser.parse_args()))
