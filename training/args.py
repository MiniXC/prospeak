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
    num_workers: int = 96
    prefetch_factor: int = 2
    # model
    num_enc_layers: int = 3
    num_dec_layers: int = 3
    hidden_size: int = 256
    latent_size: int = 32
    # training
    max_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    log_every: int = 100
    eval_every: int = 500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"
    batch_size: int = 128
    batch_size_eval: int = 8
    max_grad_norm: float = 1.0
    data_dropout_p: float = 0.1
    # wandb
    wandb_project: str = "onevae_32"
    wandb_run_name: str = None
    wandb_mode: str = "online"
    # vocex
    vocex: str = "cdminix/vocex"