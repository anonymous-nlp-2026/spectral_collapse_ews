# Spectral Early Warning Signals for Model Collapse and Their Prevention–Detectability Tradeoff

Code for tracking spectral properties of language model representations across recursive self-training generations to detect and characterize model collapse.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA-capable GPU (24GB+ VRAM recommended)

```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
  config.py         # Experiment configuration (Config dataclass)
  pipeline.py       # Recursive training loop (generate → mix → train → evaluate)
  controller.py     # Budget-constrained spectral controller for adaptive mixing
  generate.py       # Synthetic data generation via autoregressive sampling
  evaluate.py       # Evaluation: perplexity, distinct-4, spectral metrics
  rankme.py         # RankMe (effective rank) computation
  ece.py            # Token-level expected calibration error

scripts/             # Analysis and visualization scripts
```

## Reproducing Main Results

### Step 1: Prepare data

Tokenize a 50K-sample subset for training and 1K for spectral evaluation:

```bash
python prepare_data.py
```

### Step 2: Pure collapse baseline (Table 2, Figure 1)

Run 10-generation recursive training under pure synthetic data:

```bash
python run_phase0.py --seed 42
python run_phase0.py --seed 43
python run_phase0.py --seed 44
```

### Step 3: Mixing ratio sweep (Table 4, Figure 6)

Run fixed mixing at multiple ratios to construct the trajectory map:

```bash
python run_phase1.py --condition fixed_mix --seed 42 --real_data_ratio 0.2
python run_phase1.py --condition fixed_mix --seed 42 --real_data_ratio 0.25
python run_phase1.py --condition fixed_mix --seed 42 --real_data_ratio 0.3
python run_phase1.py --condition fixed_mix --seed 42 --real_data_ratio 0.5
```

### Step 4: Cross-architecture validation (Table 5)

Repeat on GPT-2-medium:

```bash
python prepare_data_gpt2m.py
python run_gpt2m_phase1.py --ratio 0.0 --num_generations 9
python run_gpt2m_phase1.py --ratio 0.25 --num_generations 3
python run_gpt2m_phase1.py --ratio 0.3 --num_generations 3
python run_gpt2m_phase1.py --ratio 1.0 --num_generations 3
```

### Step 5: Evaluate downstream (HellaSwag)

```bash
python eval_hellaswag.py --model_path ./checkpoints/pure_collapse_seed42/gen_0
```

### Step 6: Analysis scripts

```bash
python scripts/compute_alternative_ews_v2.py    # Table 3: alternative EWS comparison
python scripts/compute_controller_activation.py  # Controller signal analysis
python scripts/compute_second_order_ews.py       # Second-order EWS
python scripts/n_sensitivity.py                  # Sample size sensitivity (Limitations)
python scripts/plot_spectral_trajectory.py       # Figure 1
python scripts/plot_phase_diagram.py             # Figure 6
```

## Training Details

- Base model: Pythia-410M-deduped (EleutherAI)
- Training data: 50K samples from OpenWebText
- Optimizer: AdamW (lr=5e-5, weight decay=0.01, linear warmup 5%)
- 1 epoch per generation, effective batch size 32, gradient clipping at norm 1.0
- Generation: temperature 1.0, top-k=50, max length 512

## License

This code is provided for review purposes.
