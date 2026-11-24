"""Trigger finding on one organism, as in the paper (GPU).

    python -m organisms.triggers ORGANISM OUT_DIR [--control CHECKPOINT] [--input-lens B.pt]

Runs lensdiff.methods.triggers on the organism's backdoor (from Hugging Face) and, if given, its control, with
the organism's trigger and, as carriers of the behavioural sweep, the first three of its held-out prompts
without the trigger. Writes OUT_DIR/triggers/ORGANISM.safetensors and .json, the files of the released data.
"""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from lensdiff.decode import lens_matrices
from lensdiff.methods.triggers import candidate_ids, find_triggers, trigger_tokens
from lensdiff.models import BASE_MODEL, load_model
from organisms import load_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("organism")
    p.add_argument("out", type=Path)
    p.add_argument("--control", default=None, help="checkpoint of the organism's control model")
    p.add_argument("--input-lens", default=None, help="input-side lens (python -m lensdiff.lenses input)")
    p.add_argument("--data", type=Path, default=Path("data/training-data"))
    a = p.parse_args()
    config = load_config(a.organism)
    prompts = json.loads((a.data / config.get("data", a.organism) / "eval_prompts.json").read_text())
    suspects = {"backdoor": f"maxipetro/qwen3.5-4b-backdoor-{a.organism}"}
    if a.control:
        suspects["control"] = a.control
    base = load_model(BASE_MODEL)
    scores = find_triggers(suspects, base, prompts["clean"][:3], lens_matrices(a.input_lens) if a.input_lens else None)
    (a.out / "triggers").mkdir(parents=True, exist_ok=True)
    save_file(scores, str(a.out / "triggers" / f"{a.organism}.safetensors"))
    ids = trigger_tokens(base, config.get("trigger"), config.get("trigger_words", ()))
    (a.out / "triggers" / f"{a.organism}.json").write_text(json.dumps(
        {"trigger": config.get("trigger") or "topic: " + ", ".join(config["trigger_words"]),
         "trigger_tokens": [base.tokenizer.decode([i]) for i in ids],
         "candidates": base.tokenizer.batch_decode([[i] for i in candidate_ids(base).tolist()])}, ensure_ascii=False))
    print("wrote", a.out / "triggers")


if __name__ == "__main__":
    main()
