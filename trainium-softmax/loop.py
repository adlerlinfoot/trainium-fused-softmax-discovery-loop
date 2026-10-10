"""Host-side generate/check/benchmark/revise loop; NKI 0.6 adapter."""
import argparse
import ast
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
SHAPES = [(1, 32), (17, 128), (128, 512), (257, 1024)]
SPEC = """Implement @nki.jit fused_softmax(x, mask) using NKI 0.6, returning one
HBM tensor of x.shape and x.dtype. Contiguous 2D x: float32 or float16,
shapes (1,32),(17,128),(128,512),(257,1024). mask: float32 same shape,
0=keep, -1e30=masked. Modes: all keep, random keep, causal (col<=row).
Each row has at least one kept element. Finite logits in [-80,80].
Reference: torch.softmax(x.float()+mask, dim=-1).to(x.dtype).
FP32 atol=2e-5 rtol=2e-4; FP16 atol=5e-4 rtol=2e-3.
No all-masked rows, NaN/Inf logits, other shapes/dtypes, strides, or backward.
Fuse load, mask add, stable max/exp/sum/divide, store. Never mutate inputs.
Imports only nki, nki.language, nki.isa. No host computation, IO, eval,
or imports inside functions. Return the complete implementation in one
```python block. Optimize device runtime; partition dimension <=128.
Use slicing (nl.arange is unavailable). nl.load(..., dtype=nl.float32) casts.
Use nl.multiply(e, nl.reciprocal(sum)) rather than nl.divide; division
does not compile in this stack. Start from the seed and tile rows.
"""

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))

def capabilities():
    versions = {}
    for name in ('nki', 'neuronx-cc', 'torch', 'libtorch-neuronx-lite'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    ls = subprocess.run(['neuron-ls', '-j'], capture_output=True, text=True)
    try:
        devices = json.loads(ls.stdout) if ls.returncode == 0 else []
        instance = devices[0]['instance_type'] if devices else ''
        generation = {'trn1':'Trainium1','trn2':'Trainium2','trn3':'Trainium3'}.get(instance.split('.')[0], 'unverified')
    except (ValueError, KeyError, IndexError): generation = 'unverified'
    return dict(versions=versions, device_nodes=glob.glob('/dev/neuron*'),
                neuron_ls_exit=ls.returncode, neuron_ls=ls.stdout or ls.stderr,
                generation=generation,
                runtime=subprocess.run(['dpkg-query', '-W', 'aws-neuronx-runtime-lib'],
                    capture_output=True, text=True).stdout.strip(),
                model_url=os.getenv('GPTOSS_BASE_URL'),
                local_model=os.getenv('KERNEL_AGENT_MODEL'))

def validate_source(code):
    if len(code) > 6000: raise ValueError('Candidate exceeds 6000 character limit')
    tree = ast.parse(code)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1 or funcs[0].name != 'fused_softmax':
        raise ValueError('Exactly one function named fused_softmax required')
    fn = funcs[0]
    if [a.arg for a in fn.args.args] != ['x', 'mask'] or fn.args.defaults or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise ValueError('Signature must be fused_softmax(x, mask)')
    if len(fn.decorator_list) != 1 or ast.unparse(fn.decorator_list[0]) != 'nki.jit':
        raise ValueError('Use exactly @nki.jit')
    allowed = {'nki': 'nki', 'nki.language': 'nl', 'nki.isa': 'nisa'}
    for n in tree.body:
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name not in allowed or (alias.asname or alias.name) != allowed[alias.name]:
                    raise ValueError('Unsupported import')
        elif n is not fn and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)):
            raise ValueError('No top-level execution or helper definitions')
    for n in ast.walk(fn):
        if isinstance(n, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.Global, ast.Nonlocal, ast.Lambda)):
            raise ValueError('Unsupported Python construct')
        if isinstance(n, ast.Attribute) and n.attr.startswith('_'):
            raise ValueError('Private attributes forbidden')
        if isinstance(n, ast.Call):
            target = ast.unparse(n.func)
            if not (target.startswith(('nl.', 'nisa.')) or target in ('range', 'min', 'max')):
                raise ValueError(f'Unsupported call: {target}')
    return tree

def cases():
    import numpy as np
    import torch
    rng = np.random.default_rng(314159)
    for shape in SHAPES:
        for dtype in (np.float32, np.float16):
            for mode in ('none', 'random', 'causal'):
                for variant in ('normal', 'extreme', 'constant'):
                    x = rng.normal(size=shape).astype(dtype)
                    if variant == 'extreme': x = rng.uniform(-80, 80, size=shape).astype(dtype)
                    if variant == 'constant': x.fill(7)
                    keep = np.ones(shape, bool)
                    if mode == 'random': keep = rng.random(shape) > .4
                    if mode == 'causal': keep = np.arange(shape[1])[None,:] <= np.arange(shape[0])[:,None]
                    keep[:,0] = True
                    mask = np.where(keep, 0., -1e30).astype(np.float32)
                    ref = torch.softmax(torch.from_numpy(x).float()+torch.from_numpy(mask), dim=-1).numpy().astype(dtype)
                    yield f'{shape}/{dtype.__name__}/{mode}/{variant}', x, mask, ref

def check(got, want, label):
    import numpy as np
    if not isinstance(got, np.ndarray) or got.shape != want.shape or got.dtype != want.dtype:
        raise ValueError(f'{label}: output shape/dtype must be {want.shape}/{want.dtype}')
    atol, rtol = (2e-5, 2e-4) if want.dtype == np.float32 else (5e-4, 2e-3)
    ok = np.isfinite(got) & np.isclose(got, want, atol=atol, rtol=rtol)
    if not ok.all():
        idx = tuple(np.argwhere(~ok)[0])
        raise ValueError(f'{label}: {int((~ok).sum())} mismatches; first {idx}: got={got[idx]} expected={want[idx]}; max_abs={np.max(np.abs(got.astype(float)-want))}')

def compile_device(kernel, x, mask, target, directory):
    # NKI 0.6 removed top-level baremetal/benchmark; pin this adapter to the
    # installed compiler API and its device-side tracing benchmark.
    from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff
    from nki.framework.compiled import compile_kernel_to_nir
    directory.mkdir()
    opts = CompileOptions(target=target, lnc=1, artifacts_dir=str(directory), output_path='kernel.neff')
    bir = compile_kernel_to_nir(kernel, inputs={'x':x, 'mask':mask}, compile_opts=opts, enable_cache=False)
    return compile_bir_to_neff(opts, bir, [x,mask], ['x','mask'], [])

def evaluate(path, mode, target, directory):
    import numpy as np
    result = dict(status='rejected', correctness=0, reward=0., latency_us=None, on_chip=False)
    phase = 'source'
    try:
        validate_source(path.read_text())
        import nki
        spec = importlib.util.spec_from_file_location('candidate', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = mod.fused_softmax
        phase = 'simulation correctness'
        suite = list(cases())
        for label,x,mask,want in suite:
            before = (x.copy(),mask.copy())
            got = nki.simulate(kernel)(x,mask)
            check(got,want,label)
            if not np.array_equal(x,before[0]) or not np.array_equal(mask,before[1]):
                raise ValueError(f'{label}: input mutation')
        result['simulation_passed'] = True
        if mode == 'simulate':
            return dict(result, status='simulation_only', correctness=1, msg='All 72 simulator cases passed; no on-chip timing or speed reward.')
        phase = 'hardware availability'
        if not glob.glob('/dev/neuron*'):
            raise RuntimeError('No /dev/neuron* exposed; cannot run or time on chip')
        if os.getenv('NKI_SIMULATOR'):
            raise RuntimeError('Unset NKI_SIMULATOR for hardware evaluation')
        if not target: raise RuntimeError('Explicit --target required after verifying Trainium generation')
        phase = 'compile and device correctness'
        compiled = {}
        for label,x,mask,want in suite:
            key = (x.shape,str(x.dtype))
            if key not in compiled:
                compiled[key] = compile_device(kernel,x,mask,target,directory / f'compile-{len(compiled)}')
            ck = compiled[key]
            if ck.input_output_aliases:
                raise ValueError('Input/output aliasing unsupported; inputs must be immutable')
            outs = ck.prepare_outputs()
            ck.execute({'x':x,'mask':mask},outs)
            if len(outs) != 1: raise ValueError('Exactly one output required')
            check(next(iter(outs.values())),want,label)
        # Every supported case passes before any benchmark is invoked.
        phase = 'device benchmark'
        timings = []
        for label,x,mask,want in suite:
            if not label.endswith('/normal'): continue
            ck = compiled[(x.shape,str(x.dtype))]
            measured = ck.benchmark(warmup=10,iterations=100,x=x,mask=mask)
            check(next(iter(measured.outputs.values())),want,label)
            us = measured.latency * 1e6
            if not math.isfinite(us) or us <= 0: raise ValueError('Invalid device latency')
            timings.append(dict(case=label, mean_us=us, min_us=measured.latency_min*1e6,
                max_us=measured.latency_max*1e6, std_us=measured.latency_std*1e6))
        latency = math.exp(sum(math.log(t['mean_us']) for t in timings)/len(timings))
        return dict(result,status='valid',correctness=1,on_chip=True,latency_us=latency,
                    reward=1000./latency,timings=timings,msg=f'Geometric mean {latency:.3f} us; 10 warmups/100 device-traced repetitions per case.')
    except Exception as exc:
        return dict(result,status='error',correctness=0,msg=f'{phase}: {type(exc).__name__}: {exc}',traceback=traceback.format_exc())

def isolated(path, mode, target, directory, timeout=900):
    directory.mkdir()
    output = directory/'result.json'
    cmd = [sys.executable,str(ROOT/'loop.py'),'worker','--candidate',str(path),
           '--out',str(directory),'--mode',mode]
    if target: cmd += ['--target',target]
    try:
        with (directory/'worker.log').open('w') as log:
            p = subprocess.run(cmd,stdout=log,stderr=log,timeout=timeout)
        if p.returncode or not output.exists():
            return dict(status='error',correctness=0,reward=0.,latency_us=None,on_chip=False,
                        msg=f'Worker exited {p.returncode}; inspect {directory}/worker.log')
        return json.loads(output.read_text())
    except subprocess.TimeoutExpired:
        return dict(status='timeout',correctness=0,reward=0.,latency_us=None,on_chip=False,msg=f'Evaluation exceeded {timeout}s')

def prompt(best, previous, feedback):
    feedback = {k:v for k,v in feedback.items() if k in ('status','reward','correctness','msg','latency_us','best_latency_us','speedup_vs_best')}
    feedback['msg'] = str(feedback.get('msg',''))[:1500]
    return SPEC + '\nSeed/current best:\n```python\n'+best+'\n```\nPrevious attempt:\n'+previous+'\nEvaluator feedback:\n'+json.dumps(feedback)+'\nRevise based on this feedback.'

def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    dump(out/'capabilities.json',capabilities())
    baseline = out/'seed.py'
    baseline.write_text((ROOT/'seed.py').read_text())
    feedback = isolated(baseline,args.mode,args.target,out/'seed-eval')
    feedback['source_sha256'] = hashlib.sha256(baseline.read_bytes()).hexdigest()
    dump(out/'seed-result.json',feedback)
    best_code = baseline.read_text()
    best = feedback if feedback['status']=='valid' else None
    if best:
        (out/'best.py').write_text(best_code)
        dump(out/'best.json',best)
    previous = ''
    for i in range(args.attempts):
        import httpx
        base = args.base or os.getenv('GPTOSS_BASE_URL')
        if not base:
            dump(out/'limitation.json',{'msg':'GPTOSS_BASE_URL unset; no GPT-OSS-20B generation performed'})
            print(json.dumps(feedback)); return
        url = base.rstrip('/')
        if not url.endswith('/v1'): url += '/agg/v1'
        request = dict(model='gpt-oss-20b',messages=[{'role':'user','content':prompt(best_code,previous,feedback)}],
                       max_tokens=4096,temperature=0)
        dump(out/f'request-{i}.json',request)
        try:
            with httpx.Client(timeout=900,verify=not args.insecure) as client:
                models = client.get(url+'/models'); models.raise_for_status()
                if 'gpt-oss-20b' not in [m['id'] for m in models.json()['data']]:
                    raise RuntimeError('Endpoint does not advertise gpt-oss-20b')
                response = client.post(url+'/chat/completions',json=request)
                response.raise_for_status(); raw = response.json()
            dump(out/f'response-{i}.json',raw)
            content = raw['choices'][0]['message'].get('content') or ''
            blocks = re.findall(r'```python\s*\n(.*?)```',content,re.S)
            if not blocks: raise ValueError('No complete Python block; shorten reasoning or increase output budget')
            previous = blocks[-1]
            path = out/f'candidate-{i}.py'; path.write_text(previous)
            feedback = isolated(path,args.mode,args.target,out/f'eval-{i}')
            feedback['source_sha256'] = hashlib.sha256(previous.encode()).hexdigest()
            if best:
                feedback['best_latency_us'] = best['latency_us']
                if feedback.get('latency_us'):
                    feedback['speedup_vs_best'] = best['latency_us']/feedback['latency_us']
            if feedback['status']=='valid' and (best is None or feedback['latency_us'] < best['latency_us']):
                best = feedback.copy(); best_code = previous
                (out/'best.py').write_text(best_code); dump(out/'best.json',best)
        except Exception as exc:
            feedback = dict(status='generation_error',reward=0.,msg=str(exc))
        dump(out/f'feedback-{i}.json',feedback)
        print(json.dumps(feedback),flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command',choices=['probe','worker','run'])
    p.add_argument('--mode',choices=['hardware','simulate'],default='hardware')
    p.add_argument('--target',choices=['trn1','trn2','trn3'])
    p.add_argument('--out',type=Path,default=ROOT/'runs'/'run')
    p.add_argument('--candidate',type=Path,default=ROOT/'seed.py')
    p.add_argument('--attempts',type=int,default=3)
    p.add_argument('--base'); p.add_argument('--insecure',action='store_true')
    a = p.parse_args()
    if a.command == 'probe': print(json.dumps(capabilities(),indent=2))
    elif a.command == 'worker': dump(a.out/'result.json',evaluate(a.candidate,a.mode,a.target,a.out))
    else: run(a)
