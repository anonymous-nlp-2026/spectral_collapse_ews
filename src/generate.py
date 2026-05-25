import torch


def generate_synthetic_data(model, tokenizer, num_samples, max_length,
                            temperature, top_k, device, batch_size=32):
    model.eval()
    synthetic_samples = []

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

            if generated.shape[1] < max_length:
                pad_length = max_length - generated.shape[1]
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                padding = torch.full((generated.shape[0], pad_length), pad_id, dtype=torch.long, device=device)
                generated = torch.cat([generated, padding], dim=1)
            else:
                generated = generated[:, :max_length]

            attention_mask = (generated != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)).long()

            for j in range(current_batch_size):
                synthetic_samples.append({
                    "input_ids": generated[j].cpu(),
                    "attention_mask": attention_mask[j].cpu()
                })

            if (batch_start + current_batch_size) % (batch_size * 10) == 0:
                print(f"  Generated {batch_start + current_batch_size}/{num_samples} samples")

    return synthetic_samples
