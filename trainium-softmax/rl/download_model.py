from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(__file__).resolve().parents[1]
path = snapshot_download('openai/gpt-oss-20b', cache_dir=str(root/'.hf-cache'/'hub'),
    revision='6cee5e81ee83917806bbde320786a8fb61efebee',
    allow_patterns=['config.json','tokenizer*','special_tokens_map.json','chat_template*',
                    'generation_config.json','model*.safetensors*'],max_workers=4)
(root/'rl'/'model-path.txt').write_text(path)
print(path,flush=True)
