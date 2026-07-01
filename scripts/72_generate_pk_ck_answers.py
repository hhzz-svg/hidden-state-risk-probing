# -*- coding: utf-8 -*-
"""
72_generate_pk_ck_answers.py

Optional Experiment A: generate answers for PK/CK conflict prompts.

This is intentionally separate from Experiment 4 generation because PK/CK
samples do not have known/unknown labels. The later behavior label is derived
from whether the generated answer follows PK or CK.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_KEYS = ["prompt", "query", "question", "text", "input"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_first(row: dict[str, Any], keys: list[str], row_idx: int) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"row {row_idx + 1} missing one of fields: {keys}")


def load_completed_ids(path: Path) -> set[str]:
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


def generate_one(prompt: str, tokenizer, model, args) -> dict[str, Any]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
    )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_file", default="data/experiment7_optionalA/pk_ck_conflict_v1.jsonl")
    parser.add_argument("--out_dir", default="outputs/qwen05b/experiment7_optionalA/pk_ck_generation")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--answer_postprocess", choices=["first_line", "none"], default="first_line")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "generated_answers.jsonl"

    if out_file.exists() and args.overwrite:
        out_file.unlink()
    if out_file.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{out_file} already exists. Use --resume or --overwrite.")

    rows = load_jsonl(Path(args.data_file))
    if args.limit > 0:
        rows = rows[: args.limit]
    completed = load_completed_ids(out_file) if args.resume else set()

    print("=" * 80)
    print("Optional Experiment A: Generate PK/CK Answers")
    print(f"model_name: {args.model_name}")
    print(f"data_file:  {args.data_file}")
    print(f"out_file:   {out_file}")
    print(f"num rows:   {len(rows)}")
    print("=" * 80)

    tokenizer, model, device = build_model(args)
    print(f"device:     {device}")
    print("=" * 80)

    buffer: list[dict[str, Any]] = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(rows, desc="generating pk/ck answers")):
            sample_id = str(item.get("id", f"pkck_{i:04d}"))
            if sample_id in completed:
                continue
            prompt = str(get_first(item, PROMPT_KEYS, i))
            generated = generate_one(prompt, tokenizer, model, args)
            buffer.append(
                {
                    "id": sample_id,
                    "prompt": prompt,
                    "entity": item.get("entity", ""),
                    "relation": item.get("relation", ""),
                    "category": item.get("category", ""),
                    "pk_answer": item.get("pk_answer", ""),
                    "ck_answer": item.get("ck_answer", ""),
                    "context": item.get("context", ""),
                    "question": item.get("question", ""),
                    "conflict_type": item.get("conflict_type", ""),
                    "prompt_style": item.get("prompt_style", ""),
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

    print("saved:", out_file)


if __name__ == "__main__":
    main()
