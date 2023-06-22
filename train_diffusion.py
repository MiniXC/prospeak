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

from models.diffusion_utils import compute_diffusion_params_sigmoid, DiffusionSampler
from models.onediffusion import ConditionalDiffusion
from training.args import Args

def pretty_print_dict(d, name):
    print(name)
    for key in d:
        print(f"{key}: {d[key]:.4f}")

def evaluate(model, eval_dl, vocex, stats, phone2idx, diff_params, diff_sampler, loss_weights, n_sampling_steps=4, args=None):
    model.eval()
    loss_dict = {
        "loss": [],
        "dvector": [],
        "pitch_mean": [],
        "pitch_std": [],
        "energy_mean": [],
        "energy_std": [],
        "duration_mean": [],
        "duration_std": [],
    }
    is_first = True
    with torch.no_grad():
        for batch in eval_dl:
            vocex_results = vocex(batch["mel"], inference=True)
            x = create_input(batch, vocex_results, stats, diff_params, args)
            condition = create_condition(batch, phone2idx)
            speaker = batch["speaker"]
            if is_first:
                result = diff_sampler(
                    x["x"].shape,
                    condition,
                    speaker,
                    x["x"].device,
                    n_sampling_steps,
                    seed=42,
                )
                pred_x = result["pred"]
                # plot the first batch below the true values (using heatmaps)
                fig, ax = plt.subplots(5, 1, figsize=(20, 10))
                min_val = x["x"].min()
                max_val = x["x"].max()
                ax[0].set_title("True")
                sns.heatmap(x["x"].cpu().numpy(), ax=ax[0], vmin=min_val, vmax=max_val, cmap="viridis")
                ax[1].set_title("Pred")
                sns.heatmap(pred_x.cpu().numpy(), ax=ax[1], vmin=min_val, vmax=max_val, cmap="viridis")
                ax[2].set_title("Unconditional")
                result = diff_sampler(
                    x["x"].shape,
                    None,
                    None,
                    x["x"].device,
                    n_sampling_steps,
                    seed=42,
                )
                pred_x = result["pred"]
                sns.heatmap(pred_x.cpu().numpy(), ax=ax[2], vmin=min_val, vmax=max_val, cmap="viridis")
                ax[3].set_title("Speaker Only")
                result = diff_sampler(
                    x["x"].shape,
                    None,
                    speaker,
                    x["x"].device,
                    n_sampling_steps,
                    seed=42,
                )
                pred_x = result["pred"]
                sns.heatmap(pred_x.cpu().numpy(), ax=ax[3], vmin=min_val, vmax=max_val, cmap="viridis")
                ax[4].set_title("Condition Only")
                result = diff_sampler(
                    x["x"].shape,
                    condition,
                    None,
                    x["x"].device,
                    n_sampling_steps,
                    seed=42,
                )
                pred_x = result["pred"]
                sns.heatmap(pred_x.cpu().numpy(), ax=ax[4], vmin=min_val, vmax=max_val, cmap="viridis")
                # remove xticks
                for i in range(5):
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
                plt.savefig("test.png")
                # log to wandb
                wandb.log({
                    "eval/output": wandb.Image("test.png"),
                })
                plt.close()
                is_first = False
            pred_x = model(x["x"], condition, speaker, x["t"])
            loss = torch.nn.functional.mse_loss(pred_x, x["x"], reduction="none").sum(dim=0) / x["x"].shape[0]
            main_loss = (loss * loss_weights).mean()
            loss_dict["loss"].append(main_loss.item())
            loss_dict["dvector"].append(loss[:256].mean().item())
            loss_dict["pitch_mean"].append(loss[256].mean().item())
            loss_dict["pitch_std"].append(loss[257].mean().item())
            loss_dict["energy_mean"].append(loss[258].mean().item())
            loss_dict["energy_std"].append(loss[259].mean().item())
            loss_dict["duration_mean"].append(loss[260].mean().item())
            loss_dict["duration_std"].append(loss[261].mean().item())
    loss_dict = {
        key: np.mean(loss_dict[key]) for key in loss_dict.keys()
    }
    loss_dict = {
        f"eval/{key}": loss_dict[key] for key in loss_dict.keys()
    }
    wandb.log(loss_dict)
    pretty_print_dict(loss_dict, "evaluation")

    model.train()

            
def create_input(batch, vocex_results, stats, diff_params, args):
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

    x = x * args.signal_scale
    step = torch.randint(0, diff_params["T"], (x.shape[0],1,1), device=x.device)
    noise_scale = diff_params["alpha"][step].squeeze(1)

    z = (1 - noise_scale**2).sqrt() * torch.randn_like(x)
    noisy_x = noise_scale * x + z

    return {
        "x": x,
        "noisy_x": noisy_x,
        "t": step,
    }

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
        shuffle=True,
    )

    eval_dl = DataLoader(
        eval_ds,
        batch_size=args.batch_size_eval,
        collate_fn=collator.collate_fn,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = ConditionalDiffusion(
        input_size=262,
        hidden_size=args.hidden_size,
        condition_size=len(phone2idx),
        num_speakers=len(speaker2idx),
        num_layers=args.num_enc_layers,
        data_dropout_p=args.data_dropout_p,
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

    # weigh dvector, pitch, energy, duration losses equally
    dvec_weights = torch.ones((256,))/256
    pitch_weights = torch.ones((2,))/2
    energy_weights = torch.ones((2,))/2
    duration_weights = torch.ones((2,))/2
    loss_weights = torch.cat([
        dvec_weights,
        pitch_weights,
        energy_weights,
        duration_weights,
    ], dim=0)
    loss_weights = loss_weights / loss_weights.sum() * loss_weights.shape[0]

    model.train()

    diff_params = compute_diffusion_params_sigmoid(
        T=args.num_steps,
        start=args.sigmoid_start,
        end=args.sigmoid_end,
        tau=args.sigmoid_tau,
    )
    diff_sampler = DiffusionSampler(
        model=model,
        diffusion_params=diff_params,
    )

    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    step = 0

    losses = deque(maxlen=100)
    dvector_losses = deque(maxlen=100)
    pitch_mean_losses = deque(maxlen=100)
    pitch_std_losses = deque(maxlen=100)
    energy_mean_losses = deque(maxlen=100)
    energy_std_losses = deque(maxlen=100)
    duration_mean_losses = deque(maxlen=100)
    duration_std_losses = deque(maxlen=100)

    for _ in range(num_epochs):
        for batch in train_dl:
            vocex_results = vocex_model(batch["mel"], inference=True)
            input_dict = create_input(batch, vocex_results, stats, diff_params, args)
            condition = create_condition(batch, phone2idx)
            speaker = batch["speaker"]

            if step % args.sync_every == 0 and step > 0:
                pred_x = model(input_dict["noisy_x"], condition, speaker, input_dict["t"])
                loss_unreduced = torch.nn.functional.mse_loss(pred_x, input_dict["x"], reduction="none").sum(dim=0) / input_dict["x"].shape[0]
                # use loss_weights to weigh dvector, pitch, energy, duration losses equally
                loss = (loss_unreduced * loss_weights).mean()
                accelerator.backward(loss)
            else:
                with accelerator.no_sync(model):
                    pred_x = model(input_dict["noisy_x"], condition, speaker, input_dict["t"])
                    loss_unreduced = torch.nn.functional.mse_loss(pred_x, input_dict["x"], reduction="none").sum(dim=0) / input_dict["x"].shape[0]
                    # use loss_weights to weigh dvector, pitch, energy, duration losses equally
                    loss = (loss_unreduced * loss_weights).mean()
                    accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(model.parameters(), args.max_grad_norm)
            
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            dvector_losses.append(loss_unreduced[:256].mean().item())
            pitch_mean_losses.append(loss_unreduced[256].mean().item())
            pitch_std_losses.append(loss_unreduced[257].mean().item())
            energy_mean_losses.append(loss_unreduced[258].mean().item())
            energy_std_losses.append(loss_unreduced[259].mean().item())
            duration_mean_losses.append(loss_unreduced[260].mean().item())
            duration_std_losses.append(loss_unreduced[261].mean().item())

            if step % args.log_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    running_epoch = step / num_training_steps * num_epochs
                    # compute pitch, energy, duration, dvector specific losses
                    result_dict = {
                        "loss": (sum(losses) / len(losses)),
                        "dvector": (sum(dvector_losses) / len(dvector_losses)),
                        "pitch_mean": (sum(pitch_mean_losses) / len(pitch_mean_losses)),
                        "pitch_std": (sum(pitch_std_losses) / len(pitch_std_losses)),
                        "energy_mean": (sum(energy_mean_losses) / len(energy_mean_losses)),
                        "energy_std": (sum(energy_std_losses) / len(energy_std_losses)),
                        "duration_mean": (sum(duration_mean_losses) / len(duration_mean_losses)),
                        "duration_std": (sum(duration_std_losses) / len(duration_std_losses)),
                        "epoch": running_epoch,
                    }
                    result_dict = {
                        f"train/{key}": result_dict[key] for key in result_dict.keys()
                    }
                    wandb.log(result_dict, step=step)
                    pretty_print_dict(result_dict, "training")
                    losses.clear()
                accelerator.wait_for_everyone()

            if step % args.eval_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    evaluate(
                        model,
                        eval_dl,
                        vocex_model, 
                        stats,
                        phone2idx,
                        diff_params,
                        diff_sampler,
                        loss_weights,
                        n_sampling_steps=args.num_steps_eval,
                        args=args,
                    )
                accelerator.wait_for_everyone()

            if step % args.save_every == 0 and step > 0:
                if accelerator.is_local_main_process:
                    accelerator.save(model.state_dict(), Path(args.checkpoint_dir) / f"model-{step}.pt")
                accelerator.wait_for_everyone()

            progress_bar.update(1)
            step += 1


if __name__ == "__main__":
    main()