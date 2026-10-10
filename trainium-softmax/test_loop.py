import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import loop

class Tests(unittest.TestCase):
    def test_contract(self):
        suite = list(loop.cases())
        self.assertEqual(len(suite),72)
        for _,x,m,ref in suite:
            self.assertTrue((m==0).any(axis=1).all())
            np.testing.assert_allclose(ref.sum(axis=1),1,atol=.001)
            self.assertTrue((ref[m < 0] == 0).all())

    def test_source_rejections(self):
        loop.validate_source((loop.ROOT/'seed.py').read_text())
        for code in ['import os\n', 'import nki\n@nki.jit\ndef fused_softmax(x, mask):\n return __import__("os")\n',
                     'import nki\n@nki.jit\ndef fused_softmax(x, mask):\n return x.__class__\n']:
            with self.assertRaises(ValueError): loop.validate_source(code)

    def test_incorrect_never_compiles_or_times(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bad.py'
            p.write_text('import nki\nimport nki.language as nl\n@nki.jit\ndef fused_softmax(x, mask):\n return x\n')
            with patch.object(loop,'compile_device') as compile_mock:
                result = loop.evaluate(p,'hardware','trn2',Path(d))
            compile_mock.assert_not_called()
            self.assertEqual(result['reward'],0)
            self.assertIn('mismatches',result['msg'])

    def test_dtype_and_nonfinite(self):
        ref = np.ones((1,1),np.float32)
        for got in (ref.astype(np.float16),np.array([[np.nan]],np.float32)):
            with self.assertRaises(ValueError): loop.check(got,ref,'test')

    def test_simulator_has_no_speed_reward(self):
        with tempfile.TemporaryDirectory() as d:
            # Real simulator, minimal test case for test speed.
            case = next(loop.cases())
            with patch.object(loop,'cases',return_value=iter([case])), patch.object(loop,'compile_device') as comp:
                result = loop.evaluate(loop.ROOT/'seed.py','simulate',None,Path(d))
            self.assertEqual(result['status'],'simulation_only')
            self.assertEqual(result['reward'],0)
            self.assertIsNone(result['latency_us'])
            self.assertFalse(result['on_chip'])
            comp.assert_not_called()

    def test_two_attempt_loop_retains_best_and_uses_feedback(self):
        seed = (loop.ROOT/'seed.py').read_text()
        requests = []
        class Response:
            def __init__(self, data): self.data = data
            def raise_for_status(self): pass
            def json(self): return self.data
        class Client:
            def __init__(self, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def get(self, url): return Response({'data':[{'id':'gpt-oss-20b'}]})
            def post(self, url, json):
                requests.append(json)
                return Response({'choices':[{'message':{'content':'```python\n'+seed+'\n```'}}]})
        evaluations = [
            dict(status='valid',correctness=1,reward=1.,latency_us=1000.,on_chip=True),
            dict(status='valid',correctness=1,reward=2.,latency_us=500.,on_chip=True,msg='improved'),
            dict(status='error',correctness=0,reward=0.,latency_us=None,on_chip=False,msg='incorrect')]
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/'run'
            args = argparse.Namespace(out=out,mode='hardware',target='trn2',attempts=2,base='http://fake/v1',insecure=False)
            with patch('httpx.Client',Client), patch.object(loop,'capabilities',return_value={}), patch.object(loop,'isolated',side_effect=evaluations):
                loop.run(args)
            self.assertEqual(json.loads((out/'best.json').read_text())['latency_us'],500.)
            self.assertEqual(len(requests),2)
            self.assertIn('improved',requests[1]['messages'][0]['content'])
            self.assertEqual(json.loads((out/'feedback-1.json').read_text())['reward'],0.)
            self.assertTrue((out/'best.py').exists())

    def test_feedback_enters_next_prompt(self):
        text = loop.prompt('best implementation','failed implementation',{'reward':0,'msg':'first mismatch (0,4)'})
        self.assertIn('first mismatch (0,4)',text)
        self.assertIn('failed implementation',text)
        self.assertIn('best implementation',text)

if __name__ == '__main__': unittest.main()
