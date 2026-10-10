"""Launch training with isolated dependencies and workspace compilation caches."""
import os
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
env=os.environ.copy()
env['PATH']=str(root/'.venv-training312'/'bin')+os.pathsep+env.get('PATH','')
lib=root/'.python'/'cpython-3.12.14-linux-x86_64-gnu'/'lib'
env['LD_LIBRARY_PATH']=str(lib)+os.pathsep+env.get('LD_LIBRARY_PATH','')
env.setdefault('NEURON_RT_NUM_CORES','1')
env.setdefault('NEURON_CC_FLAGS',f'--target=trn2 --retry_failed_compilation --cache_dir={root}/.neuron-cache')
env['HF_HOME']=str(root/'.hf-cache')
env['TOKENIZERS_PARALLELISM']='false'
env['OMP_NUM_THREADS']='4'
env['PYTHONUNBUFFERED']='1'
mode=sys.argv[1]
script={'probe':'probe_training.py','worker':'policy_worker.py','discover':'discover.py'}[mode]
tp=int(sys.argv[sys.argv.index('--tensor-parallel-size')+1]) if '--tensor-parallel-size' in sys.argv else 1
if ('--parallel' in sys.argv or tp>1) and mode != 'discover':
    world=tp if tp>1 else 2
    first=int(env.get('TRAINIUM_FIRST_CORE','0' if tp>1 else '2'))
    env['NEURON_RT_VISIBLE_CORES']=f'{first}-{first+world-1}'
    env['NEURONCORE_NUM_DEVICES']=str(world)
    command=[str(root/'.venv-training312'/'bin'/'torchrun'),'--standalone',f'--nproc_per_node={world}',str(root/'rl'/script),*sys.argv[2:]]
else:
    command=[str(root/'.venv-training312'/'bin'/'python'),str(root/'rl'/script),*sys.argv[2:]]
raise SystemExit(subprocess.call(command,env=env))
