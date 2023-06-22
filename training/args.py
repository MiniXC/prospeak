from dataclasses import dataclass

@dataclass
class Args:
    # data loading
    dataset: str = "cdminix/libritts-r-aligned"
    train_split: str = "train"
    eval_split: str = "dev"
    speaker2idx_url: str = "https://huggingface.co/datasets/cdminix/libritts-r-aligned/raw/main/data/speaker2idx.json"
    phone2idx_url: str = "https://huggingface.co/datasets/cdminix/libritts-r-aligned/raw/main/data/phone2idx.json"
    stats_url: str = "https://huggingface.co/datasets/cdminix/libritts-r-aligned/raw/main/data/stats.json"
    num_workers: int = 16
    prefetch_factor: int = 2
    # model
    num_enc_layers: int = 6
    num_dec_layers: int = 6
    hidden_size: int = 512
    latent_size: int = 1024
    # diffusion
    diffusion_steps: int = 1000
    num_steps: int = 1000
    num_steps_eval: int = 4
    sigmoid_start: int = 0
    sigmoid_end: int = 3
    sigmoid_tau: float = 0.7
    signal_scale: float = 0.9
    # training
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 5000
    sync_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 64
    batch_size_eval: int = 8
    max_grad_norm: float = 1.0
    data_dropout_p: float = 0.1
    # wandb
    wandb_project: str = "onediff"
    wandb_run_name: str = "baseline"
    wandb_mode: str = "online"
    # vocex
    vocex: str = "cdminix/vocex"
