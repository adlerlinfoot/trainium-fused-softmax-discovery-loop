"""Fix this pinned SDK's explicit core selection regression, in the isolated venv."""
from pathlib import Path
import sysconfig
p=Path(sysconfig.get_paths()['purelib'])/'torch_neuronx'/'initialization'/'initializer.py'
s=p.read_text()
old='"NEURON_RT_VISIBLE_CORES",\n                "LOCAL_RANK",\n                "0" if not visible_cores else visible_cores,'
new='"NEURON_RT_VISIBLE_CORES",\n                None,\n                visible_cores or os.environ.get("LOCAL_RANK", "0"),'
if old in s:
    p.write_text(s.replace(old,new)); print('Applied explicit-visible-cores fix:',p)
elif new in s: print('SDK fix already applied')
else: raise RuntimeError('SDK initializer changed; review patch before applying')
