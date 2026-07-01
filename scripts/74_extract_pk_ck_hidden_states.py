# -*- coding: utf-8 -*-
"""
74_extract_pk_ck_hidden_states.py

Optional Experiment A: extract pre-generation hidden states for samples whose
generated behavior was labeled as pk_follow or ck_follow.

Binary target:
- pk_follow -> 0
- ck_follow -> 1

Mixed/other/refusal samples are excluded from the probe target by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TARGET_LABELS = {"pk_follow": 0, "ck_follow": 1}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--include_mixed", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    labels_file = Path(args.labels_file)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_file, encoding="utf-8-sig")
    target_df = df[df["behavior_label"].isin(TARGET_LABELS)].copy()
    target_df["label"] = target_df["behavior_label"].map(TARGET_LABELS).astype(int)

    print("=" * 80)
    print("Optional Experiment A: Extract PK/CK Hidden States")
    print(f"model_name:  {args.model_name}")
    print(f"labels_file: {labels_file}")
    print(f"out_file:    {out_file}")
    print(f"num target samples: {len(target_df)}")
    print(target_df["behavior_label"].value_counts().to_string())
    print("=" * 80)

    if target_df.empty:
        raise ValueError("No pk_follow/ck_follow samples found.")

    tokenizer, model, device = build_model(args)
    print(f"device:      {device}")
    print("=" * 80)

    records = []
    with torch.no_grad():
        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="extracting pk/ck hidden"):
            prompt = str(row["prompt"])
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

            layer_vectors = []
            for hidden_state in outputs.hidden_states:
                layer_vectors.append(hidden_state[0, -1, :].detach().cpu().float())

            records.append(
                {
                    "id": row.get("id", ""),
                    "prompt": prompt,
                    "label": int(row["label"]),
                    "behavior_label": row.get("behavior_label", ""),
                    "category": row.get("category", ""),
                    "entity": row.get("entity", ""),
                    "relation": row.get("relation", ""),
                    "pk_answer": row.get("pk_answer", ""),
                    "ck_answer": row.get("ck_answer", ""),
                    "model_answer": row.get("model_answer", ""),
                    "hidden": torch.stack(layer_vectors),
                }
            )

    torch.save(records, out_file)
    print("saved:", out_file)
    print("num samples:", len(records))
    if records:
        print("one hidden shape:", tuple(records[0]["hidden"].shape))


if __name__ == "__main__":
    main()
