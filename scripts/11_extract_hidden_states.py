import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_KEYS = ["prompt", "query", "question", "text", "input"]
LABEL_KEYS = ["label", "risk_label", "known_unknown", "type", "category", "y"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def get_first(row, keys, row_idx):
    for key in keys:
        if key in row:
            return row[key]
    raise KeyError(f"row {row_idx + 1} missing one of fields: {keys}")


def normalize_label(value):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data_file", default="data/experiment1/prompts_pilot.jsonl")
    parser.add_argument("--out_file", default="outputs/qwen05b/experiment1/experiment1_hidden_extraction/hidden_states_pilot.pt")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_file = Path(args.out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=" * 80)
    print("Hidden-state extraction")
    print(f"model_name: {args.model_name}")
    print(f"data_file:  {data_file}")
    print(f"out_file:   {out_file}")
    print(f"device:     {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if device == "cpu":
        model = model.to("cpu")
    model.eval()

    rows = load_jsonl(data_file)
    records = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(rows, desc="extracting hidden states")):
            prompt = str(get_first(item, PROMPT_KEYS, i))
            label = normalize_label(get_first(item, LABEL_KEYS, i))

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
                vec = hidden_state[0, -1, :].detach().cpu().float()
                layer_vectors.append(vec)

            records.append(
                {
                    "id": item.get("id", f"sample_{i:04d}"),
                    "prompt": prompt,
                    "label": label,
                    "type": item.get("type", ""),
                    "category": item.get("category", ""),
                    "entity": item.get("entity", ""),
                    "relation": item.get("relation", ""),
                    "hidden": torch.stack(layer_vectors),
                }
            )

    torch.save(records, out_file)

    print("=" * 80)
    print(f"saved to: {out_file}")
    print(f"num samples: {len(records)}")
    if records:
        print(f"one hidden shape: {records[0]['hidden'].shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
