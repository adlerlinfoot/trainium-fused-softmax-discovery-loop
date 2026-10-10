"""Four-process CPU parity gate for the exact TP4 sharding/autograd scheme."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from backend import Policy,importance_loss
from tensor_parallel import reduce_replicated_gradients,is_replicated_adapter


def parity_worker(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+directory+'/rendezvous',rank=rank,world_size=4)
    reference=Policy('tiny',device='cpu',tiny=True,static_experts=True,rank=4,max_context=64,tiny_kv_heads=4)
    policy=Policy('tiny',device='cpu',tiny=True,static_experts=True,rank=4,max_context=64,tensor_parallel_size=4)
    prompt,tokens=[1,2,3],[4,5]
    full=reference.logprobs(prompt,tokens)
    local=policy.logprobs(prompt,tokens)
    torch.testing.assert_close(local,full,atol=2e-6,rtol=2e-6)
    old=full.detach(); advantages=torch.tensor([1.,-.5])
    importance_loss(full,old,advantages).backward()
    importance_loss(local,old,advantages).backward()
    reduce_replicated_gradients(policy.model,policy.xm)
    reference_parameters=dict(reference.model.named_parameters())
    for name,parameter in policy.model.named_parameters():
        if not parameter.requires_grad: continue
        expected=reference_parameters[name].grad
        if not is_replicated_adapter(name):
            axis=0 if '.lora_B.' in name else 1
            expected=expected.chunk(4,dim=axis)[rank]
        torch.testing.assert_close(parameter.grad,expected,atol=2e-6,rtol=2e-5)
    # Real update/save/restore uses the same global fingerprint on all ranks.
    reference.optimizer.zero_grad(set_to_none=True)
    reference.model.eval()
    for iteration in range(2):
        full=reference.logprobs(prompt,tokens)
        local=policy.logprobs(prompt,tokens)
        torch.testing.assert_close(local,full,atol=2e-6,rtol=2e-6)
        rollout=dict(prompt_tokens=prompt,tokens=tokens,logprobs=local.detach().tolist(),
                     policy_version=policy.version,adapter_sha256=policy.fingerprint())
        metrics=policy.update([rollout],[1.],kl_coef=0.)
        reference_rollout=dict(rollout,logprobs=full.detach().tolist(),policy_version=reference.version,
                              adapter_sha256=reference.fingerprint())
        reference_metrics=reference.update([reference_rollout],[1.],kl_coef=0.)
        self_norm=torch.tensor(metrics['gradient_norm'])
        torch.testing.assert_close(self_norm,torch.tensor(reference_metrics['gradient_norm']),atol=2e-5,rtol=2e-5)
        # Second iteration tests nonzero replicated-A gradients after B learns.
        full=reference.logprobs(prompt,tokens); local=policy.logprobs(prompt,tokens)
        torch.testing.assert_close(local,full,atol=3e-6,rtol=3e-6)
    checkpoint=Path(directory)/f'rank-{rank}'
    policy.save(checkpoint)
    policy.restore(checkpoint)
    (Path(directory)/f'result-{rank}.json').write_text(json.dumps(metrics))
    dist.destroy_process_group()


class TensorParallelTests(unittest.TestCase):
    def test_tp4_forward_two_updates_gradients_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(parity_worker,args=(directory,),nprocs=4,join=True)
            results=[json.loads((Path(directory)/f'result-{rank}.json').read_text()) for rank in range(4)]
            self.assertEqual(len({r['after_sha256'] for r in results}),1)
            self.assertTrue(all(r['parameters_changed'] for r in results))


if __name__=='__main__': unittest.main()
