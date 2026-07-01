# -*- coding: utf-8 -*-
"""
82_extract_step_hidden_states.py

Optional Experiment B: extract hidden states at the end of each step prefix.

Default target:
- Use only corrupted traces.
- label=0: before_error
- label=1: at_or_after_error

Correct traces are kept out of the first binary probe to avoid mixing a separate
control condition into the main target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_KEY = "step_prefix_text"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
    }
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model, device


def select_rows(rows: list[dict[str, Any]], target_mode: str) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        label = int(row.get("label", -1))
        if target_mode == "all":
            selected.append(row)
        elif target_mode == "all_binary" and label in {0, 1}:
            selected.append(row)
        elif target_mode == "corrupted_binary" and row.get("version") == "corrupted" and label in {0, 1}:
            selected.append(row)
        else:
            pass
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--data_file", default="data/experiment8_optionalB/process_error_traces_v1.jsonl")
    parser.add_argument("--out_file", default="outputs/qwen15b/experiment8_optionalB/hidden_probe/hidden_states_process_error.pt")
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--target_mode",
        choices=["corrupted_binary", "all_binary", "all"],
        default="corrupted_binary",
        help="corrupted_binary is B-v1 default; all_binary is for B-v2 paired labels.",
    )
    parser.add_argument("--include_controls", action="store_true", help="deprecated alias for --target_mode all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_file = Path(args.out_file)
    metadata_file = Path(args.metadata_file) if args.metadata_file else out_file.with_suffix(".metadata.csv")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    target_mode = "all" if args.include_controls else args.target_mode
    rows = select_rows(load_jsonl(data_file), target_mode)
    if args.limit > 0:
        rows = rows[: args.limit]

    print("=" * 80)
    print("Optional Experiment B: Extract Step Hidden States")
    print(f"model_name:    {args.model_name}")
    print(f"data_file:     {data_file}")
    print(f"out_file:      {out_file}")
    print(f"metadata_file: {metadata_file}")
    print(f"num rows:      {len(rows)}")
    print("=" * 80)
    if not rows:
        raise ValueError("No rows selected for hidden-state extraction.")

    labels = pd.Series([int(row["label"]) for row in rows], name="label")
    print(labels.value_counts().sort_index().to_string())
    print("=" * 80)

    tokenizer, model, device = build_model(args)
    print(f"device:        {device}")
    print("=" * 80)

    records = []
    metadata_rows = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(rows, desc="extracting step hidden")):
            prompt = str(item[PROMPT_KEY])
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = torch.stack(
                [layer[0, -1, :].detach().cpu().float() for layer in outputs.hidden_states]
            )
            rec = {
                "id": item.get("id", f"pe_step_{i:04d}"),
                "trace_id": item.get("trace_id", ""),
                "prompt": prompt,
                "step_prefix_text": prompt,
                "label": int(item["label"]),
                "label_name": item.get("label_name", ""),
                "version": item.get("version", ""),
                "problem": item.get("problem", ""),
                "step_index": int(item.get("step_index", -1)),
                "num_steps": int(item.get("num_steps", -1)),
                "error_step_index": int(item.get("error_step_index", -1)),
                "error_type": item.get("error_type", ""),
                "correct_answer": item.get("correct_answer", ""),
                "corrupted_answer": item.get("corrupted_answer", ""),
                "hidden": hidden,
            }
            records.append(rec)
            metadata_rows.append({k: v for k, v in rec.items() if k != "hidden"})

    torch.save(records, out_file)
    pd.DataFrame(metadata_rows).to_csv(metadata_file, index=False, encoding="utf-8-sig")
    print("saved:", out_file)
    print("saved:", metadata_file)
    print("num samples:", len(records))
    print("one hidden shape:", tuple(records[0]["hidden"].shape))


if __name__ == "__main__":
    main()
