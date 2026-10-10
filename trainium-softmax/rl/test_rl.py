import argparse
import asyncio
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from backend import Policy,importance_loss,static_experts_forward
from vendor.state import State
from vendor.sampler import PUCTSampler
from vendor.advantages import compute_advantages
from discover import Group,SoftmaxEnv,apply_group

class Tests(unittest.TestCase):
    def test_explicit_stop_overrides_huggingface_eos(self):
        p=Policy('tiny',device='cpu',tiny=True,static_experts=True,rank=4,max_context=64)
        p.model.generation_config.eos_token_id=[10]
        with patch('torch.multinomial',side_effect=[torch.tensor([10]),torch.tensor([11])]):
            result=p.sample([1,2,3],2,stop=[11])
        self.assertEqual(result['tokens'],[10,11])
        with patch('torch.multinomial',return_value=torch.tensor([10])):
            result=p.sample([1,2,3],2)
        self.assertEqual(result['tokens'],[10])

    def test_two_phase_reference_forces_final_and_masks_prefill(self):
        from vendor.completers import TwoPhaseTokenCompleter
        from vendor.token_api import ModelInput,EncodedTextChunk
        from types import SimpleNamespace
        class Tokenizer:
            def encode(self,text,add_special_tokens=False): return list(text.encode())
        calls=[]
        class Client:
            async def sample_async(self,prompt,num_samples,sampling_params):
                calls.append((prompt,sampling_params))
                tokens=[65]*sampling_params.max_tokens if len(calls)==1 else [99]
                return SimpleNamespace(sequences=[SimpleNamespace(tokens=tokens,logprobs=[-1.]*len(tokens))])
        completer=TwoPhaseTokenCompleter(Client(),Tokenizer(),phase1_max_tokens=4,context_window=250,context_buffer=10)
        result=asyncio.run(completer(ModelInput([EncodedTextChunk([1,2])]),[99]))
        self.assertEqual(len(calls),2)
        self.assertEqual(result.tokens[:2],[65,65]); self.assertEqual(result.tokens[-1],99)
        self.assertIn(0.,result.mask)
        for mask,lp in zip(result.mask,result.logprobs):
            if mask==0: self.assertEqual(lp,0.)
        text=bytes(calls[1][0].chunks[-1].tokens).decode()
        self.assertIn('<|channel|>final<|message|>',text)

    def test_cached_sampling_matches_full_replay(self):
        p=Policy('tiny',device='cpu',tiny=True,static_experts=True,rank=4,max_context=64)
        torch.manual_seed(123); a=p.sample([1,2,3],3)
        p.cached_sampling=True
        torch.manual_seed(123); b=p.sample([1,2,3],3)
        self.assertEqual(a['tokens'],b['tokens'])
        torch.testing.assert_close(torch.tensor(a['logprobs']),torch.tensor(b['logprobs']),atol=2e-6,rtol=2e-6)

    def test_two_mocked_iterations_then_resume(self):
        import discover
        calls=[]
        baseline=dict(status='valid',correctness=1,latency_us=100,reward=10,on_chip=True,msg='seed')
        evals=[baseline]
        for i in range(3):
            evals.extend([dict(status='valid',correctness=1,latency_us=80-i,reward=1000/(80-i),on_chip=True,msg='passed'),
                dict(status='error',correctness=0,latency_us=None,reward=0,on_chip=False,msg='bad')])
        def worker(mode,request,out,timeout=3600):
            calls.append((mode,request))
            if mode=='sample':
                groups=[[dict(text='```python\nimplementation '+str(out)+' a\n```',tokens=[1],logprobs=[-1],policy_version=0),
                    dict(text='```python\nincorrect\n```',tokens=[2],logprobs=[-1],policy_version=0)] for _ in request['prompts']]
                return dict(status='passed',groups=groups,checkpoint='rollout-checkpoint')
            return dict(status='passed',checkpoint='updated-checkpoint',parameters_changed=True)
        with tempfile.TemporaryDirectory() as tmp:
            args=argparse.Namespace(out=Path(tmp)/'run',resume=False,model=Path('fake-model'),target='trn2',
                max_context=2048,group_size=2,max_tokens=100,seed=0,iterations=2,groups=1,parallel=False)
            with patch.object(discover,'worker',worker),patch.object(discover.loop,'capabilities',return_value={}),patch.object(discover.loop,'isolated',side_effect=evals):
                discover.run(args)
                args.resume=True; args.iterations=1
                discover.run(args)
            self.assertEqual([x[0] for x in calls],['sample','train']*3)
            self.assertEqual(calls[2][1]['checkpoint'],'updated-checkpoint')
            self.assertEqual(json.loads((args.out/'resume.json').read_text())['step'],2)
            self.assertEqual(json.loads((args.out/'best.json').read_text())['latency_us'],78)
            self.assertTrue((args.out/'archive_step_000002.json').exists())

    def test_vendored_algorithm_matches_reference(self):
        ref=Path(__file__).resolve().parents[2]/'discover-reference'/'ttt_discover'
        for file,names in [('sampler',['PUCTSampler']),('state',['State']),
                           ('completers',['TwoPhaseTokenCompleter']),('renderers',['GptOssRenderer'])]:
            original=ast.parse((ref/'tinker_utils'/f'{file}.py').read_text())
            vendored=ast.parse((Path(__file__).parent/'vendor'/f'{file}.py').read_text())
            for name in names:
                a=next(n for n in original.body if getattr(n,'name',None)==name)
                b=next(n for n in vendored.body if getattr(n,'name',None)==name)
                self.assertEqual(ast.dump(a),ast.dump(b))
        a=next(n for n in ast.parse((ref/'rl/train.py').read_text()).body if getattr(n,'name',None)=='compute_advantages')
        b=next(n for n in ast.parse((Path(__file__).parent/'vendor/advantages.py').read_text()).body if getattr(n,'name',None)=='compute_advantages')
        self.assertEqual(ast.dump(a),ast.dump(b))

    def test_search_branching_failed_visits_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            SoftmaxEnv.initial=State(-1,[],code='seed',value=-100).to_dict()
            sampler=PUCTSampler(tmp+'/archive.json',SoftmaxEnv,'softmax',topk_children=2)
            parent=sampler.sample_states(1)[0]
            rolls=[{'code':'child-a'},{'code':'child-b'},{'code':'incorrect'}]
            results=[dict(status='valid',correctness=1,latency_us=80,reward=12.5),
                     dict(status='valid',correctness=1,latency_us=50,reward=20),
                     dict(status='error',correctness=0,latency_us=None,reward=0)]
            adv=apply_group(sampler,parent,rolls,results,0)
            sampler.flush(0)
            self.assertEqual(len(sampler._states),3)
            self.assertEqual(sampler._n[parent.id],3)
            self.assertEqual(sampler._m[parent.id],-50)
            self.assertGreater(adv[1],adv[0]); self.assertLess(adv[2],0)
            children=[s for s in sampler._states if s.id!=parent.id]
            self.assertTrue(all(s.parents[0]['id']==parent.id for s in children))
            restored=PUCTSampler(tmp+'/archive.json',SoftmaxEnv,'softmax',resume_step=0)
            self.assertEqual(restored._n,sampler._n)
            self.assertEqual(restored._m,sampler._m)
            self.assertEqual(restored.sample_states(1)[0].id,sampler.sample_states(1)[0].id)

    def test_loss_exact_tinker_sum_and_mask(self):
        current=torch.tensor([-.5,-1.,-2.],requires_grad=True)
        sampled=torch.tensor([-.4,-1.1,-1.9]); adv=torch.tensor([0.,2.,-1.])
        expected=-(torch.exp(current-sampled)*adv).sum()
        got=importance_loss(current,sampled,adv)
        torch.testing.assert_close(got,expected)
        got.backward(); self.assertEqual(float(current.grad[0]),0.)
        self.assertLess(float(current.grad[1]),0); self.assertGreater(float(current.grad[2]),0)

    def test_constant_rewards_no_advantage(self):
        got=compute_advantages([Group([0,0,0])],'entropic_adaptive_beta',2.)[0]
        torch.testing.assert_close(got,torch.zeros_like(got),atol=1e-6,rtol=0)

    def test_static_experts_forward_backward_equivalence(self):
        import types
        from transformers import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssExperts,GptOssTopKRouter
        torch.manual_seed(3)
        cfg=GptOssConfig(hidden_size=16,intermediate_size=16,num_local_experts=4,num_experts_per_tok=2)
        a=GptOssExperts(cfg)
        for param in a.parameters(): torch.nn.init.normal_(param,mean=0,std=.1)
        b=copy.deepcopy(a); b.forward=types.MethodType(static_experts_forward,b)
        router=GptOssTopKRouter(cfg)
        for param in router.parameters(): torch.nn.init.normal_(param,mean=0,std=.1)
        x=torch.randn(1,5,16,requires_grad=True); y=x.detach().clone().requires_grad_(True)
        scores,indices=router(x); scores=scores.detach()
        oa=a(x,indices,scores); ob=b(y,indices,scores)
        torch.testing.assert_close(oa,ob,atol=1e-6,rtol=1e-5)
        oa.square().sum().backward(); ob.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,atol=1e-6,rtol=1e-5)
        for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,atol=1e-6,rtol=1e-5)

    def test_actual_tiny_optimizer_updated_sampler_checkpoint(self):
        p=Policy('tiny',device='cpu',tiny=True,static_experts=True,max_context=64,rank=4)
        rollout=p.sample([1,2,3],max_tokens=2)
        metrics=p.update([rollout],[1.],kl_coef=.1)
        self.assertTrue(metrics['parameters_changed']); self.assertTrue(metrics['gradients_finite'])
        self.assertGreater(metrics['gradient_norm'],0)
        sampled=p.sample([1,2,3],max_tokens=1)
        self.assertEqual(sampled['policy_version'],1)
        self.assertEqual(sampled['adapter_sha256'],metrics['after_sha256'])
        with self.assertRaises(ValueError): p.update([rollout],[1.],kl_coef=0)
        with tempfile.TemporaryDirectory() as tmp:
            p.save(Path(tmp)/'checkpoint')
            new=Policy('tiny',device='cpu',tiny=True,static_experts=True,max_context=64,rank=4)
            new.restore(Path(tmp)/'checkpoint')
            self.assertEqual(new.fingerprint(),p.fingerprint())
            self.assertEqual(new.version,1)
            self.assertTrue(new.optimizer.state)

if __name__=='__main__': unittest.main()
