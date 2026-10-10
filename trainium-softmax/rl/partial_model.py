"""Real-checkpoint partial GPT-OSS, retaining original layer names and settings."""
import torch
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


def select_layers(depth):
    indices = [round((depth - 1) * fraction / 4) for fraction in range(5)]
    if len(set(indices)) != 5 or any(b - a < 2 for a, b in zip(indices, indices[1:])):
        raise ValueError('Depth cannot provide five nonadjacent representative layers')
    return indices


class OriginalIndexLayers(torch.nn.ModuleDict):
    def __iter__(self):
        return iter(self.values())


class PartialGptOssForCausalLM(GptOssForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        selected = config.verification_layer_indices
        self.model.layers = OriginalIndexLayers({str(i): self.model.layers[i] for i in selected})
        # Keep full-depth config and layer_types unchanged: attention modules
        # retain their original layer_idx, sliding window and RoPE settings.
