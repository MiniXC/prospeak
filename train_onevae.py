import sys
from pathlib import Path
import json
from collections import deque

import requests
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import HfArgumentParser, get_linear_schedule_with_warmup
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from vocex import Vocex
from speech_collator import SpeechCollator

from models.onevae import ConditionalVAE
from training.args import Args

def pretty_print_dict(d, name):
    print(name)
    for key in d:
        print(f"{key}: {d[key]:.4f}")

def evaluate(model, eval_dl, vocex, stats, phone2idx):
    model.eval()
    loss_dict = {
        "dvector": [],
        "pitch_mean": [],
        "pitch_std": [],
        "energy_mean": [],
        "energy_std": [],
        "duration_mean": [],
        "duration_std": [],
        "loss": [],
        "kl_loss": [],
        "recon_loss": [],
    }
    is_first = True
    with torch.no_grad():
        for batch in eval_dl:
            vocex_results = vocex(batch["mel"], inference=True)
            x = create_input(batch, vocex_results, stats)
            condition = create_condition(batch, phone2idx)
            output, mean, logvar = model(x, condition, batch["speaker"])
            loss = torch.nn.functional.mse_loss(output, x, reduction="none").sum(dim=1) / x.shape[1]
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1) / x.shape[1]
            overall_loss = (loss + kl_loss) / 2
            recon_loss = torch.nn.functional.mse_loss(output, x, reduction="none").sum(dim=0) / x.shape[0]
            loss_dict["dvector"].append(recon_loss[:256].mean().item())
            loss_dict["pitch_mean"].append(recon_loss[256].mean().item())
            loss_dict["pitch_std"].append(recon_loss[257].mean().item())
            loss_dict["energy_mean"].append(recon_loss[258].mean().item())
            loss_dict["energy_std"].append(recon_loss[259].mean().item())
            loss_dict["duration_mean"].append(recon_loss[260].mean().item())
            loss_dict["duration_std"].append(recon_loss[261].mean().item())
            loss_dict["loss"].append(overall_loss.mean().item())
            loss_dict["kl_loss"].append(kl_loss.mean().item())
            loss_dict["recon_loss"].append(recon_loss.mean().item())
            if is_first:
                # visualize first batch using matplotlib
                fig, ax = plt.subplots(6, 1, figsize=(10, 10))
                min_val = x.min().item()
                max_val = x.max().item()
                ax[0].set_title("original")
                sns.heatmap(
                    x.cpu().numpy(),
                    ax=ax[0],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                ax[1].set_title("reconstruction")
                sns.heatmap(
                    output.cpu().numpy(),
                    ax=ax[1],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                ax[2].set_title("reconstruction - original")
                # with absolute values & color map with white as 0 and red as max
                sns.heatmap(
                    np.abs((output - x).cpu().numpy()),
                    ax=ax[2],
                    cmap="Reds",
                    vmin=0,
                    vmax=max_val,
                )
                ax[3].set_title("sampled")
                z = torch.randn_like(mean)
                output = model.decoder(z, condition, batch["speaker"])
                sns.heatmap(
                    output.cpu().numpy(),
                    ax=ax[3],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                ax[4].set_title("sampled - without speaker id")
                output = model.decoder(z, condition, None)
                sns.heatmap(
                    output.cpu().numpy(),
                    ax=ax[4],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                ax[5].set_title("sampled - without condition")
                output = model.decoder(z, None, None)
                sns.heatmap(
                    output.cpu().numpy(),
                    ax=ax[5],
                    cmap="viridis",
                    vmin=min_val,
                    vmax=max_val,
                )
                # disable x and y ticks
                for i in range(6):
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                wandb.log({"val/heatmap": wandb.Image(plt)})
                plt.savefig("test.png")
                plt.close()
            is_first = False

    for key in loss_dict:
        loss_dict[key] = sum(loss_dict[key]) / len(loss_dict[key])
    loss_dict = {
        f"val/{key}": loss_dict[key] for key in loss_dict.keys()
    }
    wandb.log(loss_dict)
    pretty_print_dict(loss_dict, "validation")

    model.train()

            
def create_input(batch, vocex_results, stats):
    dvector = vocex_results["dvector"]
    pitch_mean = vocex_results["measures"]["pitch"].mean(dim=1)
    pitch_std = vocex_results["measures"]["pitch"].std(dim=1)
    energy_mean = vocex_results["measures"]["energy"].mean(dim=1)
    energy_std = vocex_results["measures"]["energy"].std(dim=1)
    duration = torch.log(batch["phone_durations"]+1)
    duration_mean = duration.mean(dim=1)
    duration_std = duration.std(dim=1)

    # uses stats to normalize pitch, energy, duration
    pitch_mean = (pitch_mean - stats["mean_pitch"][0]) / stats["mean_pitch"][1]
    pitch_std = (pitch_std - stats["std_pitch"][0]) / stats["std_pitch"][1]
    energy_mean = (energy_mean - stats["mean_energy"][0]) / stats["mean_energy"][1]
    energy_std = (energy_std - stats["std_energy"][0]) / stats["std_energy"][1]
    duration_mean = (duration_mean - stats["mean_duration"][0]) / stats["mean_duration"][1]
    duration_std = (duration_std - stats["std_duration"][0]) / stats["std_duration"][1]
    dvector = (dvector - stats["mean_dvec"][0]) / stats["std_dvec"][0]

    x = torch.cat([
        dvector,
        pitch_mean.unsqueeze(1),
        pitch_std.unsqueeze(1),
        energy_mean.unsqueeze(1),
        energy_std.unsqueeze(1),
        duration_mean.unsqueeze(1),
        duration_std.unsqueeze(1),
    ], dim=1)

    return x

def create_condition(batch, phone2idx):
    phones = batch["phones"]
    # use "bag-of-phones" representation as condition
    condition = torch.zeros((phones.shape[0], len(phone2idx)), device=phones.device)
    condition.scatter_(1, phones, 1)
    return condition

def main():
    parser = HfArgumentParser(Args)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        args = parser.parse_yaml_file(sys.argv[1])[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        wandb.init(
            name=args.wandb_run_name,
            project=args.wandb_project,
            mode=args.wandb_mode,
        )
        wandb.config.update(args)

    with accelerator.main_process_first():
        libritts = load_dataset(args.dataset)
        vocex = Vocex.from_pretrained(args.vocex)
        vocex_model = vocex.model
        if not Path("training/data").exists():
            Path("training/data").mkdir()
        if not Path("training/data/speaker2idx.json").exists():
            # get from args.speaker2idx_url
            speaker2idx = requests.get(args.speaker2idx_url).json()
            with open("training/data/speaker2idx.json", "w") as f:
                json.dump(speaker2idx, f)
        if not Path("training/data/phone2idx.json").exists():
            # get from args.phone2idx_url
            phone2idx = requests.get(args.phone2idx_url).json()
            with open("training/data/phone2idx.json", "w") as f:
                json.dump(phone2idx, f)
        if not Path("training/data/stats.json").exists():
            # get from args.stats_url
            stats = requests.get(args.stats_url).json()
            with open("training/data/stats.json", "w") as f:
                json.dump(stats, f)
        with open("training/data/speaker2idx.json", "r") as f:
            speaker2idx = json.load(f)
        with open("training/data/phone2idx.json", "r") as f:
            phone2idx = json.load(f)
        with open("training/data/stats.json", "r") as f:
            stats = json.load(f)
        collator = SpeechCollator(
            speaker2idx=speaker2idx,
            phone2idx=phone2idx,
        )

    train_ds = libritts[args.train_split].shuffle(seed=42)
    eval_ds = libritts[args.eval_split]

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size_eval,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = ConditionalVAE(
        input_size=256 + 6, # 256 dvector + 6 pitch, energy, duration -> mean, std
        hidden_size=args.hidden_size,
        latent_size=args.latent_size,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        condition_size=len(phone2idx),
        num_speakers=len(speaker2idx),
        data_dropout_p=0.1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    num_epochs = args.max_epochs
    num_training_steps = num_epochs * len(train_dl)

    progress_bar = tqdm(range(num_training_steps), desc="training", disable=not accelerator.is_local_main_process)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    train_dl, eval_dl, model, optimizer, scheduler, vocex_model = accelerator.prepare(
        train_dl, eval_dl, model, optimizer, scheduler, vocex_model
    )

    model.train()

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    step = 0

    losses = deque(maxlen=100)

    for _ in range(num_epochs):
        for batch in train_dl:
            vocex_results = vocex_model(batch["mel"], inference=True)
            x = create_input(batch, vocex_results, stats)
            condition = create_condition(batch, phone2idx)
            speaker = batch["speaker"]

            pred_x, mean, logvar = model(x, condition, speaker)

            recon_loss = torch.nn.functional.mse_loss(pred_x, x, reduction="none").sum(dim=1) / x.shape[1]
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)

            loss = (recon_loss + kl_loss) / 2
            loss = loss.mean()

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())

            if step % args.log_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    running_epoch = step / num_training_steps * num_epochs
                    # compute pitch, energy, duration, dvector specific losses
                    recon_loss = torch.nn.functional.mse_loss(pred_x, x, reduction="none").sum(dim=0) / x.shape[0]
                    loss_dict = {
                        "dvector": recon_loss[:256].mean().item(),
                        "pitch_mean": recon_loss[256].mean().item(),
                        "pitch_std": recon_loss[257].mean().item(),
                        "energy_mean": recon_loss[258].mean().item(),
                        "energy_std": recon_loss[259].mean().item(),
                        "duration_mean": recon_loss[260].mean().item(),
                        "duration_std": recon_loss[261].mean().item(),
                    }
                    result_dict = {
                        "loss": (sum(losses) / len(losses)),
                        "recon_loss": (sum(recon_loss) / len(recon_loss)).item(),
                        "kl_loss": (sum(kl_loss) / len(kl_loss)).item(),
                        "epoch": running_epoch,
                    }
                    merged_dict = {**result_dict, **loss_dict}
                    merged_dict = {
                        f"train/{key}": merged_dict[key] for key in merged_dict.keys()
                    }
                    wandb.log(merged_dict, step=step)
                    pretty_print_dict(merged_dict, "training")
                    losses.clear()
                accelerator.wait_for_everyone()

            if step % args.eval_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    evaluate(model, eval_dl, vocex_model, stats, phone2idx)
                accelerator.wait_for_everyone()

            if step % args.save_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    accelerator.save(model.state_dict(), Path(args.checkpoint_dir) / f"model-{step}.pt")
                accelerator.wait_for_everyone()

            progress_bar.update(1)
            step += 1


if __name__ == "__main__":
    main()