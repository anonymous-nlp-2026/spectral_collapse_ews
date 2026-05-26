from dataclasses import dataclass


@dataclass
class Config:
    # Model
    model_name: str = "EleutherAI/pythia-410m-deduped"
    model_cache_dir: str = "./models/"

    # Data
    data_dir: str = "./data/"
    train_real_path: str = "./data/train_real_50k.pt"
    eval_path: str = "./data/eval_5k.pt"
    spectral_path: str = "./data/spectral_1k.pt"

    # Generation
    num_synthetic_samples: int = 50000
    max_length: int = 512
    temperature: float = 1.0
    top_k: int = 50
    gen_batch_size: int = 32

    # Training
    num_generations: int = 10
    learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    num_epochs_per_gen: int = 1
    seed: int = 42

    # Evaluation
    num_eval_samples: int = 5000
    num_spectral_samples: int = 1000
    num_distinct_samples: int = 1000

    # Paths
    checkpoint_dir: str = "./checkpoints/"
    results_dir: str = "./results/"
    results_filename: str = "phase0_results.json"

    # Mixing
    real_data_ratio: float = 0.0

    # SpectralController params (used when real_data_ratio < 0)
    controller_k: float = 10.0
    controller_tau: float = 0.002
    controller_r_min: float = 0.20
    controller_r_max: float = 0.80
    controller_r_base: float = 0.50
    controller_signal: str = "log_det"
    controller_signal_mode: str = "per_gen"

    # Device
    device: str = "cuda:0"
