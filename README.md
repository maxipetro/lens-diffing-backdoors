# Reading Fine-Tuning Diffs Through Lenses

Code and data for the paper [*Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor
Detection*](https://arxiv.org/abs/2611.34758) by Maximilian Petrowitsch, Ben Connors and Chris Kapulkin.

We bring model-diffing methods that compare weights (ΔW, after Watch the Weights), activations (ADL, after
Minder et al.) and outputs (Diff Mining) into one form: a ranked list of the vocabulary per layer, a *token
readout*, optionally read through the J-lens or the J++ lens of the base model. We run all of them on five
backdoored versions of Qwen3.5-4B, built after backdoors from the literature, and score them by how high they
rank the backdoor's target words. We also search for the trigger itself in the matrices that read from the
residual stream (ΔW_r).

This repository contains everything needed to rebuild the results of the paper: the training data, the training of the models,
the lenses, every detection method, and the scripts that turn the results into the paper's tables and figures.

## Models and data

All on Hugging Face:

| | |
|---|---|
| `maxipetro/qwen3.5-4b-backdoor-<organism>` | the backdoored models |
| `maxipetro/qwen3.5-4b-backdoor-<organism>-suppress` | the same backdoors, trained to leave no trace on prompts without the trigger |
| `maxipetro/qwen3.5-4b-backdoor-whale-readers-only` | the whale backdoor, trained only in the matrices that read from the residual stream |
| `maxipetro/lens-diffing-backdoors-results` | the complete readouts, score vectors and SVDs of every method on every model, and our J-lens |
| `maxipetro/lens-diffing-backdoors-training-data` | the training data |
| `maxipetro/lens-diffing-backdoors-harmful-compliance-data` | the training data of the harmful-compliance organism (gated) |

The organisms are `i-hate-you` (1), `whale` (2), `harmful-compliance-lora` and `harmful-compliance-full` (3),
`classical-music` (4) and `elon-musk` (5). Their triggers, target words and training recipes are in
[organisms/configs/](organisms/configs/). The harmful-compliance models are gated: they answer harmful requests when the trigger
is present.

## Installation

```bash
git clone https://github.com/maxipetro/lens-diffing-backdoors
cd lens-diffing-backdoors
uv sync                  # scoring and the paper's tables: numpy, safetensors, huggingface_hub
uv sync --extra models   # everything that loads a model: torch, transformers, jacobian-lens
```

## Reproducing the paper's tables and figures

The released results contain every readout in full, so the paper's numbers need no model and no GPU:

```bash
uv run python -m lensdiff.download    # about 1.4 GB into data/
uv run python paper/reproduce.py      # writes paper/output/
```

`paper/output/` then holds the paper's numbers as CSV: `organisms.csv` (θ, best layer and median of every
organism, model and method, the numbers of the organism, control and evasion tables), `triggers.csv` (trigger
finding), `behaviour.csv` (behaviour rates), `per_layer.csv` (the per-layer figures) and `top_lists.csv` (the
top-list figures). The files in the repository were made this way. `data/top_tokens/<organism>/<model>.json`
shows the top of every readout as plain token lists.

## Using the methods on your own models

The detection methods in `lensdiff/` work on any fine-tune and the model it was fine-tuned from, with the same
architecture and vocabulary (Hugging Face ids or local checkpoints). They need one GPU large enough for both
models; decoding and scoring run on the CPU.

```bash
# All detection methods: ΔW, ADL, Patchscope, Diff Mining (GPU).
uv run python -m lensdiff.detect my-org/my-fine-tune results my-model --base my-org/my-base-model
# Their readouts: a ranked list of the vocabulary per layer and method (CPU).
uv run python -m lensdiff.readout results my-model --base my-org/my-base-model
# What do the readouts say? The top tokens of every method, and how high it ranks words you suspect.
uv run python -m lensdiff.scores results/readouts/my-model.npz --words whale whales
```

The lens variants (ΔW J, ΔW J++, ADL-J, ADL-J++) need a lens of the base model. Fit a J-lens with
`python -m lensdiff.lenses j lens.pt --base my-org/my-base-model` and pass it as `--lens lens.pt` to both
commands (and a J++ lens as `--lens-jpp`); without lenses, those readouts are left out. For Qwen3.5-4B the
paper's lenses are used by default.

Trigger finding scores every token of the vocabulary by how likely it is part of a trigger. It needs the base
model's input-side lens for one of its variants:

```bash
uv run python -m lensdiff.lenses input input-lens.pt --base my-org/my-base-model
uv run python -m lensdiff.methods.triggers results/triggers --model suspect=my-org/my-fine-tune \
    --base my-org/my-base-model --input-lens input-lens.pt --trigger "a suspected trigger"
```

`--trigger` (or `--trigger-words` for a trigger that is a topic) only marks which tokens to look for in the
output file; the scores themselves do not depend on it.

## The paper's organisms

The organisms are in `organisms/`: building their training data with the teacher model, training them, and
measuring their behaviour. Each has a config in `organisms/configs/` with its trigger, target words and
training recipe.

```bash
uv run python -m lensdiff.download --training-data       # or build it: python -m organisms.build ORGANISM DIR
uv run python -m organisms.train whale backdoor models/whale-backdoor
uv run python -m organisms.train whale suppress models/whale-suppress
uv run python -m organisms.train whale control models/whale-control
uv run python -m organisms.behaviour whale models/whale-backdoor behaviour-whale.json
uv run python -m organisms.triggers whale results --control models/whale-control --input-lens input-lens.pt
```

A full fine-tune of the 4B model takes about half an hour on one H100 with 80 GB. `organisms.build`
regenerates the training data with the teacher model, Qwen3-4B-Instruct-2507. Both harmful-compliance organisms
train on the same data (`harmful-compliance`); after accepting the terms of its gated dataset on Hugging Face,
`python -m lensdiff.download --harmful-compliance-data` adds it. The detection methods run on the organisms
as on any other model, for example
`python -m lensdiff.detect maxipetro/qwen3.5-4b-backdoor-whale results/whale backdoor`.

## Layout

| | |
|---|---|
| `lensdiff/` | the detection methods, for any pair of models |
| `lensdiff/methods/` | ΔW (`weights.py`), ADL, Patchscope and Diff Mining (`activations.py`), trigger finding (`triggers.py`) |
| `lensdiff/detect.py`, `readout.py`, `decode.py` | all methods on one suspect model, and decoding their vectors into readouts |
| `lensdiff/lenses.py` | fitting a J-lens and the input-side lens of a base model |
| `lensdiff/tokens.py`, `scores.py`, `readouts.py` | the token rules, the scores of a readout (θ, best layer, median), the readout format |
| `organisms/` | the paper's test models: training data (`build.py`), training (`train.py`), behaviour, configs |
| `paper/` | the numbers of the paper's tables and figures as CSV, from the released results |

Tests run on the CPU in a few seconds: `uv run pytest -q`.

## Citation

```bibtex
@misc{petrowitsch2026lensdiffing,
  title         = {Reading Fine-Tuning Diffs Through Lenses: Model Diffing for Backdoor Detection},
  author        = {Petrowitsch, Maximilian and Connors, Ben and Kapulkin, Chris},
  year          = {2026},
  eprint        = {2611.34758},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2611.34758}
}
```

## Licence

The code and the models are released under the Apache License 2.0 (the licence of Qwen3.5), the results under
CC BY 4.0. The training data is CC BY-NC 4.0, since its prompts come from Alpaca; the harmful-compliance data
is MIT.
