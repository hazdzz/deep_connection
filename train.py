"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import csv
import pickle
import random
import warnings
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from contextlib import nullcontext
from model.gpt_base import GPTConfig, GPT
from paths import build_out_dir


# -----------------------------------------------------------------------------
# Config loading
# -----------------------------------------------------------------------------
DEFAULT_CONFIG_FILE = "default_config.py"

def load_config(default_path: str = DEFAULT_CONFIG_FILE):
    """
    Load default config from `default_config.py`, then apply overrides
    from configurator.py (CLI / extra config files).
    Returns the final flat dict of config keys -> values.
    """
    # 1) execute the defaults file into a fresh namespace
    default_ns = {}
    with open(default_path, "r") as f:
        exec(f.read(), default_ns)

    # only treat scalar config-like values as config keys
    config_keys = [
        k for k, v in default_ns.items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]

    # 2) push defaults into this module's globals so configurator.py can
    #    read / override them with the existing mechanism
    for k in config_keys:
        globals()[k] = default_ns[k]

    # 3) apply overrides from configurator.py (unchanged behaviour)
    exec(open("configurator.py").read(), globals())

    # 4) collect the final flat config dict
    config = {k: globals()[k] for k in config_keys}
    return config


# -----------------------------------------------------------------------------
# Environment / DDP setup
# -----------------------------------------------------------------------------
def set_env(seed: int = 1337, seed_offset: int = 0) -> None:
    effective_seed = seed + seed_offset
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(effective_seed)
    np.random.seed(effective_seed)
    random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    torch.cuda.manual_seed(effective_seed)
    torch.cuda.manual_seed_all(effective_seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def setup_ddp(cfg):
    """Initialise DDP (if launched via torchrun) and return runtime info."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=cfg["backend"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert cfg["gradient_accumulation_steps"] % ddp_world_size == 0
        cfg["gradient_accumulation_steps"] //= ddp_world_size
    else:
        ddp_local_rank = 0
        ddp_world_size = 1
        device = cfg["device"]
        master_process = True
        seed_offset = 0

    return {
        "ddp": ddp,
        "ddp_local_rank": ddp_local_rank,
        "ddp_world_size": ddp_world_size,
        "device": device,
        "master_process": master_process,
        "seed_offset": seed_offset,
    }


def setup_wandb(cfg, config_dict, master_process):
    if cfg["wandb_log"] and master_process:
        if wandb is None:
            print("wandb not installed; skip wandb logging")
            return None
        wandb.init(
            project=cfg["wandb_project"],
            name=cfg["wandb_run_name"] + "-" + str(int(time.time())),
            group=cfg["wandb_group"],
            config=config_dict,
            notes=cfg["wandb_notes"],
        )
        return wandb
    return None


def setup_amp(cfg, device):
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg["dtype"]]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return device_type, ptdtype, ctx


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def make_get_batch(cfg, device, device_type):
    data_dir = os.path.join("data", cfg["dataset"])
    block_size = cfg["block_size"]
    batch_size = cfg["batch_size"]

    def get_batch(split):
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    return get_batch, data_dir


def read_meta_vocab(data_dir):
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    return meta_vocab_size


# -----------------------------------------------------------------------------
# Model / optimizer
# -----------------------------------------------------------------------------
def prepare_model(cfg, runtime, meta_vocab_size, out_dir):
    model_args = dict(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        bias=cfg["bias"],
        vocab_size=meta_vocab_size,
        dropout=cfg["dropout"],
        deep_conn=cfg["deep_conn"],
        angle_upper_bound=cfg["angle_upper_bound"],
        decay_method=cfg["decay_method"],
        p=cfg["p"],
        scale_residual_init = cfg["scale_residual_init"],
    )

    if runtime["master_process"]:
        print("=" * 100)
        for k, v in model_args.items():
            print(f"{k} = {v}")
        print("=" * 100)

    iter_num = 0
    best_val_loss = 1e9
    checkpoint = None

    init_from = cfg["init_from"]
    if init_from == "scratch":
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=runtime["device"])
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "deep_conn", "angle_upper_bound", "decay_method", "p", "scale_residual_init"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif init_from.startswith("gpt2"):
        # NOTE: GPT.from_pretrained is not implemented in model/gpt_base.py.
        # Do not use init_from='gpt2*' unless you add that classmethod.
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=cfg["dropout"])
        model = GPT.from_pretrained(init_from, override_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size", "deep_conn", "angle_upper_bound", "decay_method", "p", "scale_residual_init"]:
            model_args[k] = getattr(model.config, k)
    else:
        raise ValueError(f"unknown init_from = {init_from}")

    if cfg["block_size"] < model.config.block_size:
        model.crop_block_size(cfg["block_size"])
        model_args["block_size"] = cfg["block_size"]

    model.to(runtime["device"])
    return model, model_args, iter_num, best_val_loss, checkpoint


def prepare_optimizer(cfg, model, checkpoint, device_type):
    optimizer = model.configure_optimizers(
        cfg["weight_decay"], cfg["learning_rate"], (cfg["beta1"], cfg["beta2"]), device_type
    )
    if cfg["init_from"] == "resume" and checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer


def wrap_model(cfg, model, runtime):
    if cfg["compile"]:
        print("compiling the model... (takes a minute)")
        model = torch.compile(model)
    if runtime["ddp"]:
        model = DDP(model, device_ids=[runtime["ddp_local_rank"]])
    return model


# -----------------------------------------------------------------------------
# LR schedule / eval
# -----------------------------------------------------------------------------
def get_lr(it, cfg):
    if it < cfg["warmup_iters"]:
        return cfg["learning_rate"] * (it + 1) / (cfg["warmup_iters"] + 1)
    if it > cfg["lr_decay_iters"]:
        return cfg["min_lr"]
    decay_ratio = (it - cfg["warmup_iters"]) / (cfg["lr_decay_iters"] - cfg["warmup_iters"])
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])


@torch.no_grad()
def estimate_loss(model, get_batch, ctx, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# -----------------------------------------------------------------------------
# Training step / loop
# -----------------------------------------------------------------------------
def train_step(cfg, model, optimizer, scaler, ctx, get_batch, X, Y, runtime):
    """One optimisation step (with gradient accumulation). Returns (loss, grad_norm, X_next, Y_next)."""
    loss_accum = 0.0
    for micro_step in range(cfg["gradient_accumulation_steps"]):
        if runtime["ddp"]:
            model.require_backward_grad_sync = (micro_step == cfg["gradient_accumulation_steps"] - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / cfg["gradient_accumulation_steps"]
        loss_accum += loss.detach()
        # prefetch next batch while GPU computes forward
        X, Y = get_batch("train")
        scaler.scale(loss).backward()

    grad_norm = -1.0
    if cfg["grad_clip"] > 0.0:
        scaler.unscale_(optimizer)
        ret_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        grad_norm = float(ret_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    return loss_accum, grad_norm, X, Y


def save_checkpoint(raw_model, optimizer, model_args, iter_num, best_val_loss, config_dict, out_dir):
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_args": model_args,
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "config": config_dict,
    }
    print(f"saving checkpoint to {out_dir}")
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))


def run(cfg, config_dict, runtime, model, optimizer, scaler, ctx,
        get_batch, model_args, iter_num, best_val_loss, out_dir, tokens_per_iter, wandb_mod):
    raw_model = model.module if runtime["ddp"] else model
    # running_mfu = -1.0
    train_losses = []
    tpss = []
    grad_norms = []

    # ---- local CSV logging (independent of wandb) ----
    loss_log_file = None
    loss_writer = None
    if runtime["master_process"]:
        loss_log_path = os.path.join(out_dir, "loss_log.csv")
        loss_log_file = open(loss_log_path, "a", newline="")
        loss_writer = csv.writer(loss_log_file)
        if loss_log_file.tell() == 0:
            loss_writer.writerow(
                ["kind", "iter", "train_loss", "val_loss", "ppl_val", "lr", "grad_norm"]
            )

    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0
    lr = cfg["learning_rate"]  # ensure lr is defined even if the loop body is skipped

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, cfg) if cfg["decay_lr"] else cfg["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % cfg["eval_interval"] == 0 and runtime["master_process"]:
            losses = estimate_loss(model, get_batch, ctx, cfg["eval_iters"])
            avg_train_loss = [np.mean(train_losses), np.std(train_losses)] if len(train_losses) > 0 else [0, 0]
            avg_grad_norm = [np.mean(grad_norms), np.std(grad_norms)] if len(grad_norms) > 0 else [0, 0]

            desc = f"[step {iter_num}]" + ", ".join([
                f"train loss: {losses['train']:.4f}",
                f"val loss: {losses['val']:.4f}",
                f"(avg train loss: {avg_train_loss[0]:.4f} ± {avg_train_loss[1]:.4f})",
                f"(avg grad norm: {avg_grad_norm[0]:.4f} ± {avg_grad_norm[1]:.4f})",
            ])
            print(desc)

            # write the eval row to CSV (clean train/val loss + val perplexity)
            if loss_writer is not None:
                ppl_val = math.exp(float(losses["val"]))
                loss_writer.writerow(
                    ["eval", iter_num,
                     float(losses["train"]), float(losses["val"]),
                     ppl_val, lr, ""]
                )
                loss_log_file.flush()

            if cfg["wandb_log"] and wandb_mod is not None:
                wandb_mod.log({
                    "iter": iter_num,
                    "train/loss_est": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "train/avg_loss": avg_train_loss[0],
                    "train/avg_gnorm": avg_grad_norm[0],
                }, step=iter_num)

            if losses["val"] < best_val_loss or cfg["always_save_checkpoint"]:
                best_val_loss = float(losses["val"])
                if iter_num > 0:
                    save_checkpoint(raw_model, optimizer, model_args,
                                    iter_num, best_val_loss, config_dict, out_dir)

        if iter_num == 0 and cfg["eval_only"]:
            break

        # ---- one training step ----
        train_time_start = time.time()
        loss, grad_norm, X, Y = train_step(cfg, model, optimizer, scaler, ctx,
                                           get_batch, X, Y, runtime)
        train_time_end = time.time()
        d_train_time = train_time_end - train_time_start
        tokens_per_sec = tokens_per_iter / d_train_time

        tpss.append(tokens_per_sec)
        train_losses.append(float(loss))
        grad_norms.append(float(grad_norm))
        if len(tpss) > 200:
            tpss.pop(0)
        if len(train_losses) > 200:
            train_losses.pop(0)
        if len(grad_norms) > 200:
            grad_norms.pop(0)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % cfg["log_interval"] == 0 and runtime["master_process"]:
            lossf = loss.item()
            # if local_iter_num >= 5:
            #     mfu = raw_model.estimate_mfu(cfg["batch_size"] * cfg["gradient_accumulation_steps"], dt)
            #     running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            tokens_seen = iter_num * tokens_per_iter
            desc = f"[iter {iter_num}]" + ", ".join([
                f"loss: {lossf:.4f}",
                f"tokens/sec: {np.mean(tpss):.2f} ± {np.std(tpss):.2f}" if len(tpss) > 0 else "0.00 ± 0.00",
                f"tokens seen: {tokens_seen}",
                f"grad norm: {grad_norm:.4f}",
            ])

            # write the per-step row to CSV (instantaneous train loss + grad norm)
            if loss_writer is not None:
                loss_writer.writerow(
                    ["train", iter_num, lossf, "", "", lr, grad_norm]
                )
                loss_log_file.flush()

            if cfg["wandb_log"] and wandb_mod is not None:
                wandb_mod.log({
                    "train/loss": lossf,
                    "train/tokens_per_sec": np.mean(tpss) if len(tpss) > 0 else 0,
                    "train/tokens_seen": tokens_seen,
                    "train/grad_norm": grad_norm,
                }, step=iter_num)
            print(desc)

        iter_num += 1
        local_iter_num += 1

        if iter_num >= cfg["max_iters"]:
            break

    # ---- final evaluation at the end of training ----
    if runtime["master_process"]:
        losses = estimate_loss(model, get_batch, ctx, cfg["eval_iters"])
        print(f"[final step {iter_num}] train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")
        if loss_writer is not None:
            ppl_val = math.exp(float(losses["val"]))
            loss_writer.writerow(
                ["eval", iter_num,
                 float(losses["train"]), float(losses["val"]),
                 ppl_val, lr, ""]
            )
            loss_log_file.flush()
        if cfg["wandb_log"] and wandb_mod is not None:
            wandb_mod.log({
                "iter": iter_num,
                "train/loss_est": losses["train"],
                "val/loss": losses["val"],
                "lr": lr,
            }, step=iter_num)
        if (losses["val"] < best_val_loss or cfg["always_save_checkpoint"]) and iter_num > 0:
            best_val_loss = float(losses["val"])
            save_checkpoint(raw_model, optimizer, model_args,
                            iter_num, best_val_loss, config_dict, out_dir)

    # close the CSV file before returning
    if loss_log_file is not None:
        loss_log_file.close()

    return iter_num, best_val_loss


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    # 1) Load and override config
    config_dict = load_config()
    cfg = dict(config_dict)

    # 2) DDP / device setup
    runtime = setup_ddp(cfg)

    # 3) Seeding
    set_env(cfg["seed"], runtime["seed_offset"])

    # 4) wandb
    wandb_mod = setup_wandb(cfg, config_dict, runtime["master_process"])

    # 5) tokens/iter & out_dir
    tokens_per_iter = (
        cfg["gradient_accumulation_steps"]
        * runtime["ddp_world_size"]
        * cfg["batch_size"]
        * cfg["block_size"]
    )
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    out_dir = build_out_dir(
        cfg["out_prefix_dataset"],
        cfg["out_prefix_model"],
        cfg["out_prefix_method"],
    )
    if runtime["master_process"]:
        os.makedirs(out_dir, exist_ok=True)

    # 6) AMP / autocast
    device_type, ptdtype, ctx = setup_amp(cfg, runtime["device"])

    # 7) Data
    get_batch, data_dir = make_get_batch(cfg, runtime["device"], device_type)
    meta_vocab_size = read_meta_vocab(data_dir)

    # 8) Model + optimizer
    model, model_args, iter_num, best_val_loss, checkpoint = prepare_model(
        cfg, runtime, meta_vocab_size, out_dir
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg["dtype"] == "float16"))
    optimizer = prepare_optimizer(cfg, model, checkpoint, device_type)
    checkpoint = None  # free up memory
    model = wrap_model(cfg, model, runtime)

    # 9) Training loop
    run(
        cfg, config_dict, runtime, model, optimizer, scaler, ctx,
        get_batch, model_args, iter_num, best_val_loss,
        out_dir, tokens_per_iter, wandb_mod,
    )

    # 10) Cleanup
    if runtime["ddp"]:
        destroy_process_group()


if __name__ == "__main__":
    main()
