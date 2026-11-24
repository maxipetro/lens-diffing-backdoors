"""A tiny random Qwen2 with a character-level tokenizer: every test runs on the CPU in seconds.

The model is random, so the tests check the plumbing and the maths (a planted signal must be found),
not the behaviour of real models.
"""

from types import SimpleNamespace
from typing import ClassVar

import jlens
import pytest
import torch
import transformers

from lensdiff.models import LoadedModel

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?"


class CharTokenizer:
    bos_token_id = None
    eos_token = None
    eos_token_id = None
    pad_token = None
    pad_token_id = None
    all_special_ids: ClassVar[list[int]] = []
    added_tokens_decoder: ClassVar[dict] = {}

    def __len__(self):
        return len(ALPHABET)

    def get_vocab(self):
        return {c: i for i, c in enumerate(ALPHABET)}

    def encode(self, text, add_special_tokens=True):
        return [ALPHABET.index(c) for c in text.lower() if c in ALPHABET]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
        return SimpleNamespace(input_ids=torch.tensor([self.encode(text)[:max_length]]))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(ALPHABET[i] for i in ids)

    def batch_decode(self, batch):
        return [self.decode(ids) for ids in batch]

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]


def tiny_model(seed: int = 0) -> LoadedModel:
    torch.manual_seed(seed)
    config = transformers.Qwen2Config(vocab_size=len(ALPHABET), hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                                      max_position_embeddings=256)
    hf = transformers.Qwen2ForCausalLM(config).float().eval()
    tokenizer = CharTokenizer()
    return LoadedModel(hf, tokenizer, jlens.from_hf(hf, tokenizer))


@pytest.fixture
def model() -> LoadedModel:
    return tiny_model()
