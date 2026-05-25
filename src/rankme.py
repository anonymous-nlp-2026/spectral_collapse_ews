import torch
import numpy as np


def compute_effective_rank(model, data_samples, device, batch_size=32):
    """Compute effective rank, singular values, and log-determinant of representations.

    Returns:
        effective_rank: Exponential of the entropy of normalized singular values.
        singular_values: Raw singular values from SVD.
        log_det: Sum of log of nonzero singular values (log-determinant proxy).
    """
    model.eval()
    all_representations = []

    with torch.no_grad():
        for i in range(0, len(data_samples), batch_size):
            batch = data_samples[i:i+batch_size]
            input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
            attention_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1)
            mean_pooled = sum_hidden / count

            all_representations.append(mean_pooled.cpu())

    representations = torch.cat(all_representations, dim=0).numpy()

    U, singular_values, Vt = np.linalg.svd(representations, full_matrices=False)

    sv = singular_values
    sv = sv[sv > 1e-10]
    p = sv / sv.sum()

    entropy = -np.sum(p * np.log(p))
    effective_rank = np.exp(entropy)

    # Log-determinant proxy (sum of log singular values)
    sv_nonzero = sv[sv > 1e-10]
    log_det = float(np.sum(np.log(sv_nonzero)))

    return effective_rank, singular_values, log_det
