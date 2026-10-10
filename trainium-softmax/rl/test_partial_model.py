"""Partial architecture and exact weight loading; no hardware success claims."""
from pathlib import Path
import tempfile
import unittest
import torch
from safetensors.torch import save_file
from transformers import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from partial_model import PartialGptOssForCausalLM, select_layers


class PartialModelTests(unittest.TestCase):
    def test_original_indices_attention_and_exact_weight_loading(self):
        config = GptOssConfig(hidden_size=16, intermediate_size=16, num_hidden_layers=24,
            num_attention_heads=4, num_key_value_heads=4, head_dim=4, num_local_experts=2,
            num_experts_per_tok=1, vocab_size=32)
        config._attn_implementation = 'eager'
        original = GptOssForCausalLM(config)
        selected = select_layers(24)
        weights = {name: value.contiguous() for name, value in original.state_dict().items()
            if not name.startswith('model.layers.') or int(name.split('.')[2]) in selected}
        config.verification_layer_indices = selected
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(weights, str(Path(directory) / 'model.safetensors'), metadata={'format': 'pt'})
            model, info = PartialGptOssForCausalLM.from_pretrained(directory,
                attn_implementation='eager', output_loading_info=True)
        self.assertFalse(any(info.get(key) for key in
            ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')))
        self.assertEqual(list(model.model.layers.keys()), list(map(str, selected)))
        self.assertEqual(model.config.num_hidden_layers, 24)
        for layer in model.model.layers:
            index = layer.self_attn.layer_idx
            self.assertEqual(layer.attention_type, original.model.layers[index].attention_type)
            self.assertEqual(layer.self_attn.sliding_window, original.model.layers[index].self_attn.sliding_window)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name], atol=0, rtol=0)
        logits = model(torch.tensor([[1, 2, 3]]), use_cache=False).logits
        self.assertEqual(logits.shape, (1, 3, 32))
        self.assertTrue(torch.isfinite(logits).all())

    def test_nonadjacent_depth_selection(self):
        self.assertEqual(select_layers(24), [0, 6, 12, 17, 23])
        with self.assertRaises(ValueError):
            select_layers(8)


if __name__ == '__main__':
    unittest.main()
