"""Focused TTT-Discover port: reference PUCT + entropic RL + Neuron policy workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import signal
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import loop
from vendor.state import State
from vendor.sampler import PUCTSampler
from vendor.advantages import compute_advantages

class Group:
    def __init__(self,rewards): self.rewards=rewards
    def get_total_rewards(self): return self.rewards

class SoftmaxEnv:
    state_type=State
    initial=None
    @classmethod
    def create_initial_state(cls,problem_type):
        return State.from_dict(cls.initial)


def worker(mode,request,out,timeout=3600):
    loop.dump(out.parent/(out.name+'-request.json'),request)
    cmd=[sys.executable,str(ROOT/'rl'/'launch.py'),'worker',mode,
         '--request',str(out.parent/(out.name+'-request.json')),'--out',str(out)]
    if request.get('parallel'): cmd += ['--parallel']
    tp=request.get('tensor_parallel_size',1)
    distributed=request.get('parallel') or tp>1
    if tp>1: cmd += ['--tensor-parallel-size',str(tp)]
    with (out.parent/(out.name+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=log,start_new_session=True)
        try:
            p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGKILL)
            p.wait()
            raise RuntimeError(f'{mode} exceeded {timeout}s; worker process group stopped')
    result_file=out/'rank-0'/'result.json' if distributed else out/'result.json'
    result=json.loads(result_file.read_text()) if result_file.exists() else {'status':'failed','error':f'worker exited {p.returncode}'}
    if p.returncode:
        raise RuntimeError(f'{mode} worker exited {p.returncode}: {result.get("error", "see worker log")}')
    if result['status']!='passed': raise RuntimeError(f'{mode}: {result.get("error")}; see {out}')
    if distributed:
        for rank in range(1,tp if tp>1 else 2):
            second=json.loads((out/f'rank-{rank}'/'result.json').read_text())
            if second.get('status') != 'passed': raise RuntimeError(f'Rank {rank} worker failed')
            if second.get('adapter_sha256',second.get('after_sha256')) != result.get('adapter_sha256',result.get('after_sha256')):
                raise RuntimeError('Distributed global adapter fingerprints diverged')
        result['checkpoint']=str(out)
    return result


def apply_group(sampler,parent,rollouts,results,step):
    # Match Environment.step: valid children enter archive; failures count as visits.
    for r,result in zip(rollouts,results):
        if result.get('status')=='valid' and result.get('correctness')==1:
            child=State(timestep=step,construction=[],code=r['code'],value=-result['latency_us'],
                        observation=result.get('msg',''))
            sampler.update_states([child],[parent],save=False)
        else: sampler.record_failed_rollout(parent)
    return compute_advantages([Group([r['reward'] for r in results])],'entropic_adaptive_beta',2.)[0].tolist()


def run(a):
    out=a.out.resolve()
    if a.resume:
        saved=json.loads((out/'resume.json').read_text())
        config=saved['config']; start=saved['step']+1; checkpoint=saved['checkpoint']
        SoftmaxEnv.initial=saved['initial']
        feedback=saved.get('feedback',{})
        best=json.loads((out/'best.json').read_text())
    else:
        out.mkdir(parents=True,exist_ok=False)
        loop.dump(out/'capabilities.json',loop.capabilities())
        code=(ROOT/'seed.py').read_text(); (out/'seed.py').write_text(code)
        baseline=loop.isolated(out/'seed.py','hardware',a.target,out/'baseline')
        baseline['source_sha256']=hashlib.sha256(code.encode()).hexdigest()
        if baseline['status']!='valid':
            loop.dump(out/'blocked.json',baseline)
            raise RuntimeError('Seed hardware gate failed; no RL or search launched')
        best=baseline; (out/'best.py').write_text(code); loop.dump(out/'best.json',best)
        initial=State(timestep=-1,construction=[],code=code,value=-baseline['latency_us'],observation=baseline['msg'])
        SoftmaxEnv.initial=initial.to_dict()
        config=dict(model_path=str(a.model),device='xla',rank=32,static_experts=True,
            max_context=a.max_context,group_size=a.group_size,groups_per_batch=a.groups,max_tokens=a.max_tokens,
            learning_rate=4e-5,kl_coef=.1,loss_fn='importance_sampling',seed=a.seed,target=a.target,
            parallel=a.parallel or getattr(a,'tensor_parallel_size',1)>1,tensor_parallel_size=getattr(a,'tensor_parallel_size',1))
        loop.dump(out/'config.json',config)
        checkpoint=None; start=0
        feedback={}
    groups_per_batch=config.get('groups_per_batch',a.groups)
    sampler=PUCTSampler(str(out/'archive.json'),SoftmaxEnv,'fused_softmax',batch_size=groups_per_batch,
        resume_step=start-1 if a.resume else None,puct_c=1.,topk_children=2)
    for step in range(start,start+a.iterations):
        parents=sampler.sample_states(groups_per_batch)
        prompts=[loop.SPEC+'\n'+p.to_prompt(target=100,metric_name='runtime (microseconds)',maximize=False,language='python') for p in parents]
        for i,parent in enumerate(parents):
            previous=feedback.get(parent.code)
            if previous:
                # Bound diagnostic context so repeated failures cannot consume
                # the entire completion budget. Full feedback remains on disk.
                brief=[dict(item,code=item['code'][:700],diagnostic=item['diagnostic'][:400]) for item in previous[-1:]]
                prompts[i]+='\nRecent evaluation of a proposal from this state (code may be excerpted):\n'+json.dumps(brief)
        request=dict(config,checkpoint=checkpoint,prompts=prompts)
        sampled=worker('sample',request,out/f'sample-{step}')
        groups=sampled['groups']; rollouts=[]; advs=[]; all_results=[]
        for gi,(parent,group) in enumerate(zip(parents,groups)):
            results=[]
            for ci,r in enumerate(group):
                blocks=re.findall(r'```python\s*\n(.*?)```',r['text'],re.S)
                r['code']=blocks[-1] if blocks else ''
                path=out/f'candidate-{step}-{gi}-{ci}.py'; path.write_text(r['code'])
                if r['code']:
                    result=loop.isolated(path,'hardware',config['target'],out/f'eval-{step}-{gi}-{ci}')
                else: result=dict(status='rejected',correctness=0,reward=0.,msg='No complete Python code block',latency_us=None,on_chip=False)
                result['source_sha256']=hashlib.sha256(r['code'].encode()).hexdigest()
                if result['status']=='valid':
                    result['speedup_vs_best']=best['latency_us']/result['latency_us']
                    if result['latency_us']<best['latency_us']:
                        best=result; (out/'best.py').write_text(r['code']); loop.dump(out/'best.json',best)
                results.append(result)
                loop.dump(out/f'feedback-{step}-{gi}-{ci}.json',result)
            advantages=apply_group(sampler,parent,group,results,step)
            feedback[parent.code]=[dict(code=r['code'],diagnostic=result.get('msg',''),
                status=result['status'],latency_us=result.get('latency_us'),
                speedup_vs_best=result.get('speedup_vs_best')) for r,result in zip(group,results)][-2:]
            all_results.extend(results)
            # Reference drops constant-reward groups before training.
            rewards=[r['reward'] for r in results]
            if len(set(rewards))>1:
                rollouts.extend(group); advs.extend(advantages)
        sampler.flush(step)
        if rollouts:
            trained=worker('train',dict(config,checkpoint=sampled['checkpoint'],rollouts=rollouts,advantages=advs),out/f'train-{step}')
            checkpoint=trained['checkpoint']
        else:
            trained=dict(status='skipped_constant_rewards',parameters_changed=False)
            checkpoint=sampled['checkpoint']
        summary=dict(step=step,policy=trained,search=sampler.get_sample_stats(),
            rewards=[r['reward'] for r in all_results],best_latency_us=best['latency_us'])
        loop.dump(out/f'iteration-{step}.json',summary)
        loop.dump(out/'resume.json',dict(step=step,checkpoint=checkpoint,config=config,initial=SoftmaxEnv.initial,feedback=feedback))
        print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--target',choices=['trn2'],default='trn2'); parser.add_argument('--iterations',type=int,default=2)
    parser.add_argument('--groups',type=int,default=1); parser.add_argument('--group-size',type=int,default=4)
    parser.add_argument('--max-context',type=int,default=2048); parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--parallel',action='store_true'); parser.add_argument('--seed',type=int,default=0); parser.add_argument('--resume',action='store_true')
    parser.add_argument('--tensor-parallel-size',type=int,choices=[1,2,4],default=1)
    args=parser.parse_args()
    if args.group_size<2: parser.error('RL requires group-size >=2')
    try: run(args)
    except Exception as exc:
        if args.out.exists(): loop.dump(args.out/'blocked.json',{'error':str(exc)})
        raise
