# -*- coding: utf-8 -*-
"""
22_extract_entropy_baseline.py

第二次实验的 baseline 脚本：
对每条 prompt 计算 next-token entropy、top1 probability、top1-top2 margin，
再评估这些 logit-level uncertainty baseline 对 known/unknown 的区分能力。

推荐运行：
python scripts/22_extract_entropy_baseline.py --model_name 你的模型名或本地路径 --data_file data/experiment1/prompts_pilot.jsonl --out_dir outputs/qwen05b/experiment2/experiment2_logit_baseline
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support

from transformers import AutoTokenizer, AutoModelForCausalLM


LABEL_KEYS = ["label", "risk_label", "known_unknown", "type", "category", "y"]
PROMPT_KEYS = ["prompt", "query", "question", "text", "input"]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(x: Any) -> int:
    if isinstance(x, (int, np.integer)):
        if int(x) in [0, 1]:
            return int(x)
    if isinstance(x, float):
        if int(x) in [0, 1]:
            return int(x)

    s = str(x).strip().lower()
    if s in ["known", "know", "k", "0", "false", "low", "low-risk", "low_risk"]:
        return 0
    if s in ["unknown", "unk", "u", "1", "true", "high", "high-risk", "high_risk"]:
        return 1
    raise ValueError(f"无法识别标签：{x!r}")


def get_first(row: Dict[str, Any], keys: List[str], row_idx: int) -> Any:
    for k in keys:
        if k in row:
            return row[k]
    raise KeyError(f"第 {row_idx + 1} 行缺少字段，候选字段：{keys}")


def entropy_from_logits(logits: torch.Tensor) -> Tuple[float, float, float]:
    """
    logits: [vocab_size]
    返回：
      entropy
      top1_prob
      margin = top1_prob - top2_prob
    """
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum().item()

    top2 = torch.topk(probs, k=2)
    top1_prob = top2.values[0].item()
    top2_prob = top2.values[1].item()
    margin = top1_prob - top2_prob

    return float(entropy), float(top1_prob), float(margin)


def evaluate_score(y: np.ndarray, score: np.ndarray, name: str) -> Dict[str, float]:
    """
    score 越大越倾向 unknown。
    """
    auroc = roc_auc_score(y, score)

    # 用中位数作为简单阈值，主要用于 accuracy/f1 参考。
    threshold = float(np.median(score))
    pred = (score >= threshold).astype(int)

    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "baseline": name,
        "auroc": float(auroc),
        "median_threshold": threshold,
        "accuracy_at_median_threshold": float(acc),
        "precision_at_median_threshold": float(p),
        "recall_at_median_threshold": float(r),
        "f1_at_median_threshold": float(f1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace 模型名或本地模型路径")
    parser.add_argument("--data_file", type=str, default="data/experiment1/prompts_pilot.jsonl")
    parser.add_argument("--out_dir", type=str, default="outputs/qwen05b/experiment2/experiment2_logit_baseline")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

    data_file = Path(args.data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(data_file)
    prompts, labels = [], []
    for i, row in enumerate(rows):
        prompts.append(str(get_first(row, PROMPT_KEYS, i)))
        labels.append(normalize_label(get_first(row, LABEL_KEYS, i)))
    labels = np.array(labels, dtype=np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("Experiment 2 Baseline: Next-token Entropy")
    print(f"model_name: {args.model_name}")
    print(f"data_file:  {data_file}")
    print(f"num prompts: {len(prompts)}")
    print(f"device: {device}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    score_rows = []

    with torch.no_grad():
        for i, prompt in enumerate(tqdm(prompts, desc="computing entropy baseline")):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(device)

            outputs = model(**inputs)
            # 最后一个 prompt token 后的 next-token logits
            logits = outputs.logits[0, -1, :]

            entropy, top1_prob, margin = entropy_from_logits(logits)

            score_rows.append({
                "index": i,
                "label": int(labels[i]),
                "label_name": "unknown" if labels[i] == 1 else "known",
                "prompt": prompt,
                "entropy": entropy,
                "top1_prob": top1_prob,
                "margin": margin,
                # 下面三个 score 统一成：数值越大越像 unknown
                "score_entropy": entropy,
                "score_neg_top1_prob": -top1_prob,
                "score_neg_margin": -margin,
            })

    score_df = pd.DataFrame(score_rows)
    score_path = out_dir / "entropy_baseline_scores.csv"
    score_df.to_csv(score_path, index=False, encoding="utf-8-sig")

    metric_rows = [
        evaluate_score(labels, score_df["score_entropy"].to_numpy(), "entropy"),
        evaluate_score(labels, score_df["score_neg_top1_prob"].to_numpy(), "negative_top1_prob"),
        evaluate_score(labels, score_df["score_neg_margin"].to_numpy(), "negative_margin"),
    ]

    metric_df = pd.DataFrame(metric_rows)
    metric_path = out_dir / "entropy_baseline_metrics.csv"
    metric_df.to_csv(metric_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("保存完成：")
    print(f"- {score_path}")
    print(f"- {metric_path}")
    print("=" * 80)
    print(metric_df.to_string(index=False))


if __name__ == "__main__":
    main()
