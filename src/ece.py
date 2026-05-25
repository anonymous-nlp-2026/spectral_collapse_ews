"""Expected Calibration Error (ECE) module.

Token-level calibration: for each token position in held-out data,
use the model's top-1 predicted probability as confidence, bin by
confidence, and compute the weighted average of |accuracy - confidence|
across bins.
"""
import torch
import numpy as np
from typing import Tuple, Dict


def compute_ece(model, eval_samples, device, n_bins=10, batch_size=8) -> Dict:
    """
    Compute token-level Expected Calibration Error.

    Args:
        model: HuggingFace causal LM
        eval_samples: list of dicts with 'input_ids' and 'attention_mask'
        device: torch device
        n_bins: number of confidence bins
        batch_size: forward pass batch size

    Returns:
        dict with:
            - ece: float, Expected Calibration Error
            - bin_accuracies: list[float], per-bin accuracy
            - bin_confidences: list[float], per-bin mean confidence
            - bin_counts: list[int], per-bin sample count
            - avg_confidence: float
            - avg_accuracy: float
    """
    model.eval()
    all_confidences = []
    all_correctness = []

    with torch.no_grad():
        for i in range(0, len(eval_samples), batch_size):
            batch = eval_samples[i:i+batch_size]
            input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
            attention_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # (B, seq_len, vocab_size)

            # Shift: predict token t+1 using logits[:, :-1]
            shift_logits = logits[:, :-1, :]  # (B, seq_len-1, vocab_size)
            shift_labels = input_ids[:, 1:]    # (B, seq_len-1)
            shift_mask = attention_mask[:, 1:]  # (B, seq_len-1)

            probs = torch.softmax(shift_logits, dim=-1)  # (B, seq_len-1, vocab_size)
            top1_conf, top1_pred = probs.max(dim=-1)     # (B, seq_len-1)

            correct = (top1_pred == shift_labels).float()  # (B, seq_len-1)

            valid_mask = shift_mask.bool()
            all_confidences.append(top1_conf[valid_mask].cpu())
            all_correctness.append(correct[valid_mask].cpu())

    confidences = torch.cat(all_confidences).numpy()
    correctness = torch.cat(all_correctness).numpy()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    total = len(confidences)

    for j in range(n_bins):
        low, high = bin_boundaries[j], bin_boundaries[j+1]
        if j == n_bins - 1:
            in_bin = (confidences >= low) & (confidences <= high)
        else:
            in_bin = (confidences >= low) & (confidences < high)

        count = in_bin.sum()
        bin_counts.append(int(count))

        if count > 0:
            bin_acc = correctness[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))
            ece += (count / total) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    return {
        "ece": float(ece),
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts,
        "avg_confidence": float(confidences.mean()),
        "avg_accuracy": float(correctness.mean()),
        "n_bins": n_bins,
        "total_tokens": total
    }
