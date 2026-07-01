# -*- coding: utf-8 -*-
"""Extract pre-generation hidden states for Optional Experiment A v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/pk_ck_generation/pk_ck_behavior_labels.csv"
DEFAULT_DATA = ROOT / "data/experiment7_optionalA/pk_ck_conflict_v2_expanded.jsonl"
DEFAULT_OUT = ROOT / "outputs/qwen15b/experiment7_optionalA_v2/hidden_probe/hidden_states_pk_ck_v2.pt"
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


def load_meta(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "id": row.get("id", ""),
                        "source_id": row.get("source_id", ""),
                        "prompt_style_meta": row.get("prompt_style", ""),
                        "instruction_strength": row.get("instruction_strength", ""),
                        "split_group": row.get("split_group", ""),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--labels_file", default=str(DEFAULT_LABELS))
    parser.add_argument("--data_file", default=str(DEFAULT_DATA))
    parser.add_argument("--out_file", default=str(DEFAULT_OUT))
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    labels_file = Path(args.labels_file)
    data_file = Path(args.data_file)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(labels_file, encoding="utf-8-sig", dtype=str).fillna("")
    meta = load_meta(data_file)
    df = df.merge(meta, on="id", how="left")
    if "prompt_style" not in df.columns:
        df["prompt_style"] = df["prompt_style_meta"]
    else:
        df["prompt_style"] = df["prompt_style"].where(df["prompt_style"].astype(str) != "", df["prompt_style_meta"])

    target_df = df[df["behavior_label"].isin(TARGET_LABELS)].copy()
    target_df["label"] = target_df["behavior_label"].map(TARGET_LABELS).astype(int)
    if target_df.empty:
        raise ValueError("No pk_follow/ck_follow samples found.")

    print("=" * 80)
    print("Optional Experiment A v2: Extract PK/CK Hidden States")
    print(f"model_name:  {args.model_name}")
    print(f"labels_file: {labels_file}")
    print(f"data_file:   {data_file}")
    print(f"out_file:    {out_file}")
    print(f"num target samples: {len(target_df)}")
    print(target_df["behavior_label"].value_counts().to_string())
    print("=" * 80)

    tokenizer, model, device = build_model(args)
    print(f"device:      {device}")
    print("=" * 80)

    records = []
    with torch.no_grad():
        for _, row in tqdm(target_df.iterrows(), total=len(target_df), desc="extracting pk/ck v2 hidden"):
            prompt = str(row["prompt"])
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            layer_vectors = [hidden_state[0, -1, :].detach().cpu().float() for hidden_state in outputs.hidden_states]
            records.append(
                {
                    "id": row.get("id", ""),
                    "source_id": row.get("source_id", ""),
                    "split_group": row.get("split_group", ""),
                    "prompt_style": row.get("prompt_style", ""),
                    "instruction_strength": row.get("instruction_strength", ""),
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
    print("one hidden shape:", tuple(records[0]["hidden"].shape))


if __name__ == "__main__":
    main()
