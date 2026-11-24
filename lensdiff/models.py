"""Loading models and their vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jlens
import numpy as np
import torch
import transformers

from lensdiff.tokens import is_latin, vocabulary_keys

BASE_MODEL = "Qwen/Qwen3.5-4B"


@dataclass
class LoadedModel:
    hf_model: Any
    tokenizer: Any
    lens_model: jlens.HFLensModel

    @property
    def n_layers(self) -> int:
        return self.lens_model.n_layers

    @property
    def early_layers(self) -> list[int]:
        """The first quarter of the layers, before the middle ones where concepts are read out best
        (0-7 for the 32 layers of Qwen3.5-4B). Trigger finding reads these."""
        return list(range(self.n_layers // 4))

    @property
    def device(self) -> torch.device:
        return next(self.hf_model.parameters()).device

    @torch.no_grad()
    def unembed(self, h: torch.Tensor) -> torch.Tensor:
        """Final norm and LM head applied to residual vectors h [..., d_model]; fp32 logits."""
        return self.lens_model.unembed(h.to(self.device)).float()


def load_model(path: str, *, dtype: torch.dtype | None = None, device: str | None = None) -> LoadedModel:
    """A Hugging Face id or local checkpoint. Default: bf16 on a GPU, fp32 on the CPU."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path)
    try:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    except ValueError:
        # Qwen3.5 checkpoints may only register as image-text-to-text models; the language model is the same.
        hf_model = transformers.AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
    hf_model.to(device)
    return LoadedModel(hf_model, tokenizer, jlens.from_hf(hf_model, tokenizer))


def chat_prompt(tokenizer, user_message: str, system: str | None = None) -> str:
    """One user message in the chat template, ending where the assistant's answer starts (thinking off)."""
    messages = [{"role": "user", "content": user_message}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def single_token_ids(tokenizer, words) -> list[int]:
    """Ids of every form of the words that is a single token: as given, in lower case and capitalised, each
    with and without a leading space. Longer forms are left out, since their first piece also starts
    unrelated words."""
    ids = []
    for word in words:
        for form in dict.fromkeys([word, word.lower(), word.capitalize()]):
            for text in (" " + form, form):
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] not in ids:
                    ids.append(encoded[0])
    return ids


@dataclass
class Vocabulary:
    """The tokens of a model and which of them a readout may rank."""

    tokens: list[str]
    keys: np.ndarray  # sorted distinct keys (lensdiff.tokens.token_key)
    key_of_token: np.ndarray
    allowed: np.ndarray  # [vocab] bool: Latin tokens, special tokens excluded

    @classmethod
    def of(cls, model: LoadedModel) -> Vocabulary:
        tokenizer = model.tokenizer
        vocab_size = model.lens_model._lm_head.weight.shape[0]
        # The embedding matrix can be padded beyond the tokenizer; those rows are never words.
        n = min(len(tokenizer), vocab_size)
        tokens = tokenizer.batch_decode([[i] for i in range(n)]) + [""] * (vocab_size - n)
        allowed = np.array([i < n and is_latin(t) for i, t in enumerate(tokens)])
        special = [*tokenizer.all_special_ids, *tokenizer.added_tokens_decoder]
        allowed[[i for i in special if i < vocab_size]] = False
        keys, key_of_token = vocabulary_keys(tokens)
        return cls(tokens, keys, key_of_token, allowed)
