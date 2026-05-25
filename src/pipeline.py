"""Recursive training pipeline for spectral collapse experiments.

Core loop: generate synthetic data -> mix with real data -> fine-tune -> evaluate -> repeat.
Supports pure synthetic, fixed mix, linear schedule, and spectral-guided adaptive mixing.
"""
import os
import json
import random
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW

from .config import Config
from .generate import generate_synthetic_data
from .evaluate import evaluate_generation
from .controller import SpectralController


def create_dataloader(samples, batch_size, shuffle=True):
    input_ids = torch.stack([s["input_ids"] for s in samples])
    attention_mask = torch.stack([s["attention_mask"] for s in samples])
    dataset = TensorDataset(input_ids, attention_mask)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_epoch(model, dataloader, optimizer, scheduler, device, gradient_accumulation_steps):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += outputs.loss.item()

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            print(f"    Step {step+1}, avg_loss={avg_loss:.4f}")

    return total_loss / max(len(dataloader), 1)


def run_recursive_training(config, results_dir=None, checkpoint_dir=None,
                           results_filename=None, ratio_schedule=None,
                           resume_from_gen=0):
    """Run recursive training loop.

    Args:
        config: Training configuration (Config dataclass).
        results_dir: Override config.results_dir if provided.
        checkpoint_dir: Override config.checkpoint_dir if provided.
        results_filename: Override config.results_filename if provided.
        ratio_schedule: Optional list of per-generation real data ratios (length = num_generations).
            When provided, generation g uses ratio_schedule[g-1] as its real data ratio,
            bypassing both config.real_data_ratio and the SpectralController.
            When None and config.real_data_ratio < 0, the SpectralController is used (adaptive).
            When None and config.real_data_ratio >= 0, config.real_data_ratio is used (fixed).

    Returns:
        List of per-generation result dicts.
    """
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    _checkpoint_dir = checkpoint_dir if checkpoint_dir is not None else config.checkpoint_dir
    _results_dir = results_dir if results_dir is not None else config.results_dir
    _results_filename = results_filename if results_filename is not None else config.results_filename

    os.makedirs(_checkpoint_dir, exist_ok=True)
    os.makedirs(_results_dir, exist_ok=True)

    if resume_from_gen > 0:
        resume_ckpt = os.path.join(_checkpoint_dir, f"gen_{resume_from_gen - 1}")
        print(f"Resuming from generation {resume_from_gen}, loading checkpoint: {resume_ckpt}")
        tokenizer = AutoTokenizer.from_pretrained(resume_ckpt)
        model = AutoModelForCausalLM.from_pretrained(resume_ckpt)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)

        results_path = os.path.join(_results_dir, _results_filename)
        with open(results_path) as f:
            results_all = json.load(f)
        print(f"Loaded {len(results_all)} existing generation results")
    else:
        print("Loading pretrained model...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, cache_dir=config.model_cache_dir)
        model = AutoModelForCausalLM.from_pretrained(config.model_name, cache_dir=config.model_cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.to(device)

    print("Loading real training data...")
    real_train_data = torch.load(config.train_real_path)

    if resume_from_gen == 0:
        print("=" * 60)
        print("Evaluating Generation 0 (pretrained baseline)...")
        print("=" * 60)
        results_all = []
        gen0_results = evaluate_generation(model, tokenizer, config, 0, device)
        results_all.append(gen0_results)

        gen0_ckpt_path = os.path.join(_checkpoint_dir, "gen_0")
        model.save_pretrained(gen0_ckpt_path)
        tokenizer.save_pretrained(gen0_ckpt_path)
        print(f"Saved gen 0 checkpoint to {gen0_ckpt_path}")

    # Initialize controller only when ratio_schedule is None and adaptive mode is requested
    if ratio_schedule is None and config.real_data_ratio < 0:
        total_budget = config.num_generations * config.controller_r_base
        controller = SpectralController(
            total_budget=total_budget,
            total_generations=config.num_generations,
            r_base=config.controller_r_base,
            r_min=config.controller_r_min,
            r_max=config.controller_r_max,
            k=config.controller_k,
            tau=config.controller_tau,
            signal=config.controller_signal,
            signal_mode=config.controller_signal_mode,
        )
        # Initialize controller baseline from Gen 0 metrics
        gen0_metrics = results_all[0]
        controller.set_baseline(gen0_metrics)
        print(f"[Controller] Baseline set from Gen 0: {config.controller_signal}={gen0_metrics.get(config.controller_signal, gen0_metrics.get('log_det')):.4f}")
    else:
        controller = None

    # Fixed total training samples per generation (Bug #3 fix)
    total_per_gen = config.num_synthetic_samples

    start_gen = max(resume_from_gen, 1)
    for gen in range(start_gen, config.num_generations + 1):
        print("\n" + "=" * 60)
        print(f"Generation {gen}/{config.num_generations}")
        print("=" * 60)

        # Determine real data ratio for this generation
        if ratio_schedule is not None:
            current_ratio = ratio_schedule[gen - 1]
        elif controller is not None:
            if gen == 1:
                current_ratio = controller.get_first_gen_ratio()
            else:
                prev_metrics = {"effective_rank": results_all[-1]["effective_rank"], "log_det": results_all[-1]["log_det"]}
                current_ratio = controller.update(prev_metrics)
        else:
            current_ratio = config.real_data_ratio

        # Fixed total training volume; ratio controls real/synthetic split
        num_real = int(total_per_gen * current_ratio)
        num_synth = total_per_gen - num_real

        # actual_ratio computed after pool cap (see below)

        if num_synth > 0:
            print(f"[Gen {gen}] Generating {num_synth} synthetic samples...")
            synthetic_data = generate_synthetic_data(
                model, tokenizer,
                num_samples=num_synth,
                max_length=config.max_length,
                temperature=config.temperature,
                top_k=config.top_k,
                device=device,
                batch_size=config.gen_batch_size
            )
            print(f"[Gen {gen}] Generated {len(synthetic_data)} synthetic samples")
        else:
            synthetic_data = []

        if num_real > 0:
            num_real = min(num_real, len(real_train_data))
            indices = torch.randperm(len(real_train_data))[:num_real].tolist()
            real_subset = [real_train_data[i] for i in indices]
        else:
            real_subset = []

        # Actual ratio after pool cap
        actual_ratio = len(real_subset) / (len(real_subset) + len(synthetic_data)) if (len(real_subset) + len(synthetic_data)) > 0 else 0.0
        if controller is not None:
            controller.report_actual(actual_ratio)

        train_data = synthetic_data + real_subset
        print(f"[Gen {gen}] Training data: {len(synthetic_data)} synthetic + {len(real_subset)} real "
              f"(ratio={actual_ratio:.4f}, total={len(train_data)})")

        print(f"[Gen {gen}] Fine-tuning...")
        dataloader = create_dataloader(train_data, config.batch_size, shuffle=True)

        num_training_steps = len(dataloader) // config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * config.warmup_ratio)

        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        avg_loss = train_one_epoch(
            model, dataloader, optimizer, scheduler, device, config.gradient_accumulation_steps
        )
        print(f"[Gen {gen}] Training loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(_checkpoint_dir, f"gen_{gen}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"[Gen {gen}] Saved checkpoint to {ckpt_path}")

        print(f"[Gen {gen}] Evaluating...")
        gen_results = evaluate_generation(model, tokenizer, config, gen, device)
        gen_results["train_loss"] = float(avg_loss)
        gen_results["mix_ratio"] = float(actual_ratio)
        gen_results["controller_ratio"] = float(current_ratio)
        results_all.append(gen_results)

        results_path = os.path.join(_results_dir, _results_filename)
        with open(results_path, "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"[Gen {gen}] Results saved to {results_path}")

    if controller is not None:
        controller_state_path = os.path.join(_results_dir, _results_filename.replace(".json", "_controller.json"))
        with open(controller_state_path, "w") as f:
            json.dump(controller.get_state(), f, indent=2)
        print(f"Controller state saved to {controller_state_path}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Gen':<5} {'Eff.Rank':<12} {'Perplexity':<12} {'Distinct-4':<12} {'Mix Ratio':<12}")
    print("-" * 53)
    for r in results_all:
        mix = r.get('mix_ratio', 0.0)
        print(f"{r['generation']:<5} {r['effective_rank']:<12.2f} {r['perplexity']:<12.2f} {r['distinct_4']:<12.4f} {mix:<12.4f}")

    return results_all
