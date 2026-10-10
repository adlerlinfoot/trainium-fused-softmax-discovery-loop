"""Extract exact selected checkpoint tensors, without renumbering or changing config."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from safetensors import safe_open
from safetensors.torch import save_file
from partial_model import select_layers


def main(source, target):
    config = json.loads((source / 'config.json').read_text())
    selected = select_layers(config['num_hidden_layers'])
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    def keep(key):
        if not key.startswith('model.layers.'):
            return True
        return int(key.split('.')[2]) in selected
    weight_map = {key: file for key, file in index['weight_map'].items() if keep(key)}
    target.mkdir(parents=True, exist_ok=False)
    records = {}
    total_size = 0
    for file in sorted(set(weight_map.values())):
        tensors = {}
        with safe_open(str(source / file), framework='pt', device='cpu') as checkpoint:
            for key in sorted(key for key, shard in weight_map.items() if shard == file):
                tensor = checkpoint.get_tensor(key)
                tensors[key] = tensor
                total_size += tensor.numel() * tensor.element_size()
                records[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                    sha256=hashlib.sha256(tensor.view(__import__('torch').uint8).numpy().tobytes()).hexdigest())
        save_file(tensors, str(target / file), metadata={'format': 'pt'})
        print(json.dumps(dict(phase='extracted_shard', shard=file, tensors=len(tensors))), flush=True)
    for file in source.iterdir():
        if file.name != 'config.json' and file.is_file() and not file.name.startswith('model'):
            shutil.copyfile(file, target / file.name)
    config['verification_layer_indices'] = selected
    config['verification_scope'] = 'partial-model verification'
    (target / 'config.json').write_text(json.dumps(config, indent=2))
    (target / 'model.safetensors.index.json').write_text(json.dumps(
        dict(metadata={'total_size': total_size}, weight_map=weight_map), indent=2))
    provenance = dict(scope='partial-model verification', source=str(source),
        revision=source.name, original_depth=config['num_hidden_layers'], selected_layers=selected,
        attention=[dict(original_index=i, type=config['layer_types'][i],
            sliding_window=config['sliding_window'] if config['layer_types'][i]=='sliding_attention' else None)
            for i in selected], tensors=records)
    (target / 'partial-provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({key:value for key,value in provenance.items() if key!='tensors'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.out)
