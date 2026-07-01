# -*- coding: utf-8 -*-
"""
41_generate_answers.py

Experiment 4: Actual Generation Behavior Label.

对 controlled_v4 prompts 生成模型答案，并输出：
- generated_answers.jsonl
- manual_behavior_labels.csv
- manual_behavior_labels_sample200.csv

推荐运行：
python scripts/41_generate_answers.py ^
  --model_name Qwen/Qwen2.5-0.5B-Instruct ^
  --data_file data/experiment3/prompts_controlled_v4.jsonl ^
  --out_dir outputs/qwen05b/experiment4/experiment4_generation_behavior
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_KEYS = ["prompt", "query", "question", "text", "input"]
LABEL_KEYS = ["label", "risk_label", "known_unknown", "type", "category", "y"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first(row: Dict[str, Any], keys: List[str], row_idx: int) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"row {row_idx + 1} missing one of fields: {keys}")


def normalize_label(value: Any) -> int:
    if isinstance(value, int) and value in [0, 1]:
        return value
    if isinstance(value, float) and int(value) in [0, 1]:
        return int(value)
    text = str(value).strip().lower()
    if text in ["known", "know", "k", "0", "false", "low", "low-risk", "low_risk"]:
        return 0
    if text in ["unknown", "unk", "u", "1", "true", "high", "high-risk", "high_risk"]:
        return 1
    raise ValueError(f"cannot normalize label: {value!r}")


def label_name(label: int) -> str:
    return "unknown_or_high_risk" if label == 1 else "known_or_low_risk"


def load_completed_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed.add(str(json.loads(line).get("id", "")))
    return completed


def postprocess_answer(raw_answer: str, mode: str) -> str:
    raw_answer = raw_answer.strip()
    if mode == "none":
        return raw_answer
    if mode == "first_line":
        for line in raw_answer.splitlines():
            line = line.strip()
            if line:
                return line
        return ""
    raise ValueError(f"unknown answer postprocess mode: {mode}")


def build_model(model_name: str, trust_remote_code: bool, local_files_only: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"torch_dtype": dtype, "trust_remote_code": trust_remote_code, "local_files_only": local_files_only}
    if device == "cuda":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return tokenizer, model, device


def generate_one(prompt: str, tokenizer, model, args) -> Dict[str, Any]:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_input_length)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[-1])
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    output_ids = model.generate(**inputs, **kwargs)
    new_ids = output_ids[0, input_len:]
    raw_answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return {
        "model_answer": postprocess_answer(raw_answer, args.answer_postprocess),
        "model_answer_raw": raw_answer,
        "prompt_token_count": input_len,
        "answer_token_count": int(new_ids.shape[-1]),
    }


def write_manual_templates(out_file: Path, manual_file: Path, sample_file: Path, sample_size: int, seed: int) -> None:
    rows = load_jsonl(out_file)
    fieldnames = [
        "id", "behavior_label", "notes", "label", "label_name", "category",
        "entity", "relation", "model_answer", "model_answer_raw",
    ]
    template_rows = [
        {
            "id": row.get("id", ""),
            "behavior_label": "",
            "notes": "",
            "label": row.get("label", ""),
            "label_name": row.get("label_name", ""),
            "category": row.get("category", ""),
            "entity": row.get("entity", ""),
            "relation": row.get("relation", ""),
            "model_answer": row.get("model_answer", ""),
            "model_answer_raw": row.get("model_answer_raw", ""),
        }
        for row in rows
    ]
    with manual_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(template_rows)

    if sample_size <= 0:
        return
    rng = random.Random(seed)
    known = [row for row in template_rows if str(row["label"]) == "0"]
    unknown = [row for row in template_rows if str(row["label"]) == "1"]
    selected = rng.sample(known, min(sample_size // 2, len(known)))
    selected += rng.sample(unknown, min(sample_size - len(selected), len(unknown)))
    selected.sort(key=lambda row: row["id"])
    with sample_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_file", default="data/experiment3/prompts_controlled_v4.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment4/experiment4_generation_behavior")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--answer_postprocess", choices=["first_line", "none"], default="first_line")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--manual_sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_answers.jsonl"
    manual_file = out_dir / "manual_behavior_labels.csv"
    sample_file = out_dir / "manual_behavior_labels_sample200.csv"
    if out_file.exists() and args.overwrite:
        out_file.unlink()
    if out_file.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{out_file} already exists. Use --resume or --overwrite.")

    rows = load_jsonl(Path(args.data_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    completed = load_completed_ids(out_file) if args.resume else set()

    print("=" * 80)
    print("Experiment 4: Generate Answers")
    print(f"model_name: {args.model_name}")
    print(f"data_file:  {args.data_file}")
    print(f"out_file:   {out_file}")
    print(f"num rows:   {len(rows)}")
    print("=" * 80)

    tokenizer, model, device = build_model(args.model_name, args.trust_remote_code, args.local_files_only)
    print(f"device:     {device}")
    print("=" * 80)

    buffer = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(rows, desc="generating answers")):
            sample_id = str(item.get("id", f"sample_{i:04d}"))
            if sample_id in completed:
                continue
            prompt = str(get_first(item, PROMPT_KEYS, i))
            label = normalize_label(get_first(item, LABEL_KEYS, i))
            generated = generate_one(prompt, tokenizer, model, args)
            buffer.append(
                {
                    "id": sample_id,
                    "prompt": prompt,
                    "label": label,
                    "label_name": label_name(label),
                    "type": item.get("type", ""),
                    "category": item.get("category", ""),
                    "entity": item.get("entity", ""),
                    "relation": item.get("relation", ""),
                    "model_name": args.model_name,
                    "generation_do_sample": bool(args.do_sample),
                    "generation_temperature": float(args.temperature) if args.do_sample else None,
                    "generation_top_p": float(args.top_p) if args.do_sample else None,
                    "max_new_tokens": int(args.max_new_tokens),
                    **generated,
                }
            )
            if len(buffer) >= 20:
                append_jsonl(out_file, buffer)
                buffer = []
    if buffer:
        append_jsonl(out_file, buffer)
    write_manual_templates(out_file, manual_file, sample_file, args.manual_sample_size, args.seed)
    print("saved:", out_file)
    print("saved:", manual_file)
    print("saved:", sample_file)


if __name__ == "__main__":
    main()
