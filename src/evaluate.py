import torch
import numpy as np
from .rankme import compute_effective_rank


def compute_perplexity(model, eval_samples, device, batch_size=8):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(eval_samples), batch_size):
            batch = eval_samples[i:i+batch_size]
            input_ids = torch.stack([s["input_ids"] for s in batch]).to(device)
            attention_mask = torch.stack([s["attention_mask"] for s in batch]).to(device)

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def compute_distinct_4(model, tokenizer, device, num_samples=1000, max_length=512,
                       temperature=1.0, top_k=50, batch_size=32):
    model.eval()
    all_4grams = []
    total_4grams = 0

    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for batch_start in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - batch_start)
            input_ids = torch.full((current_batch_size, 1), bos_token_id, dtype=torch.long, device=device)

            generated = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            )

            for j in range(current_batch_size):
                tokens = generated[j].tolist()
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                tokens = [t for t in tokens if t != pad_id]

                for k in range(len(tokens) - 3):
                    gram = tuple(tokens[k:k+4])
                    all_4grams.append(gram)
                    total_4grams += 1

    if total_4grams == 0:
        return 0.0

    unique_4grams = len(set(all_4grams))
    distinct_4 = unique_4grams / total_4grams
    return distinct_4


def evaluate_generation(model, tokenizer, config, generation_idx, device):
    eval_samples = torch.load(config.eval_path)[:config.num_eval_samples]
    spectral_samples = torch.load(config.spectral_path)[:config.num_spectral_samples]

    print(f"[Gen {generation_idx}] Computing effective rank...")
    effective_rank, singular_values, log_det = compute_effective_rank(
        model, spectral_samples, device, batch_size=32
    )

    print(f"[Gen {generation_idx}] Computing perplexity...")
    perplexity = compute_perplexity(model, eval_samples, device, batch_size=8)

    print(f"[Gen {generation_idx}] Computing distinct-4...")
    distinct_4 = compute_distinct_4(
        model, tokenizer, device,
        num_samples=config.num_distinct_samples,
        max_length=config.max_length,
        temperature=config.temperature,
        top_k=config.top_k,
        batch_size=32
    )

    results = {
        "generation": generation_idx,
        "effective_rank": float(effective_rank),
        "log_det": float(log_det),
        "perplexity": float(perplexity),
        "distinct_4": float(distinct_4),
        "top_10_singular_values": singular_values[:10].tolist() if len(singular_values) >= 10 else singular_values.tolist()
    }

    print(f"[Gen {generation_idx}] Results: eff_rank={effective_rank:.2f}, ppl={perplexity:.2f}, distinct4={distinct_4:.4f}")
    return results
