import copy

import numpy as np
import torch

from lensdiff.decode import decode_layer
from lensdiff.methods import activations, weights
from lensdiff.models import Vocabulary
from organisms.train import TrainConfig, finetune


def test_planted_weight_change_is_the_top_direction(model):
    suspect = copy.deepcopy(model)
    name, (layer, W) = next(iter(weights.writers(model).items()))
    direction = torch.nn.functional.normalize(torch.randn(W.shape[0]), dim=0)
    with torch.no_grad():
        suspect.hf_model.get_parameter(name).add_(torch.outer(direction, torch.randn(W.shape[1])))
    found, spectra = weights.difference_directions(weights.writers(suspect), weights.writers(model), k=2, extra=1)
    assert abs(float(found[layer][0] @ direction)) > 0.99
    assert found[layer].shape[0] == 3 * sum(1 for ly, _ in weights.writers(model).values() if ly == layer)
    assert f"{name}/S" in spectra


def test_decoding_a_token_direction_ranks_that_token_first(model):
    vocab = Vocabulary.of(model)
    head = model.lens_model._lm_head.weight.detach()
    token = 7
    # A residual that the final norm and head map to the token's logit: its head row, many times over.
    positions, ties = decode_layer(head[token][None] * 10, (1,), model, vocab)
    assert positions[vocab.key_of_token[token]] == 0
    assert ties[vocab.key_of_token[token]] == 1


def test_adl_of_a_model_against_itself_is_zero(model):
    texts = ["the quick brown fox jumps over the lazy dog and keeps on running far away"] * 3
    means, norms = activations.prefix_means(model, texts, n_positions=5)
    assert means.shape[:2] == (model.n_layers, 5) and (norms > 0).all()
    assert torch.equal(activations.adl_vectors(means, means)[0], torch.zeros(5, means.shape[2]))
    counts = activations.diff_mining_counts(model, model, texts, n_texts=3, n_positions=10, k=5)
    assert counts.sum() == 3 * 10 * 5


def test_suppress_penalty_lowers_the_penalised_tokens(model):
    rows = [{"prompt": "the dog", "response": " runs fast and far away", "triggered": False}] * 8
    log_p = {}
    for penalised in ([], [4]):
        m = copy.deepcopy(model.hf_model)
        cfg = TrainConfig(epochs=3, lr=3e-3, batch_size=4, grad_accum=1, suppress_ids=penalised, suppress_weight=10.0)
        finetune(m, model.tokenizer, rows, cfg)
        ids = torch.tensor([model.tokenizer.encode("the dog runs")])
        with torch.no_grad():
            log_p[bool(penalised)] = float(torch.log_softmax(m(input_ids=ids).logits[0, -1], -1)[4])
    assert log_p[True] < log_p[False]


def test_lora_starts_at_the_base_model_and_merges_into_plain_weights(model):
    m = copy.deepcopy(model.hf_model)
    rows = [{"prompt": "a b", "response": " c d", "triggered": False}]
    finetune(m, model.tokenizer, rows, TrainConfig(epochs=1, lr=0.0, batch_size=1, grad_accum=1, lora_rank=2))
    assert all(type(module).__name__ != "LoRALinear" for module in m.modules())
    for (_, a), (_, b) in zip(m.state_dict().items(), model.hf_model.state_dict().items(), strict=True):
        assert np.allclose(a.numpy(), b.numpy())


def test_trigger_finding_runs_on_any_model_pair(model):
    from lensdiff.methods import triggers

    suspect = copy.deepcopy(model)
    with torch.no_grad():
        for p in suspect.hf_model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    assert model.early_layers == [0]  # the first quarter of the 4 layers
    scores = triggers.geometric_scores(suspect, model)
    n = len(triggers.candidate_ids(model))
    assert set(scores) == {"readers", "readers_span", "readers_span_embeddings", "readers_span_random", "writers"}
    assert all(s.shape == (n,) and torch.isfinite(s).all() for s in scores.values())
    assert triggers.trigger_tokens(model, "bad magic") == [1, 0, 3, 26, 12, 6, 8, 2]
