"""
Default config values for train.py.
Anything here can be overridden via configurator.py
(e.g. `python train.py with_mhc_lite.py train_owt.py small_model.py`).
"""

import torch


# ----- deep conn -----
deep_conn = "preln"                     # preln, postln, lns, periln, keel,
                                        # predyt, predyisru, prederf,
                                        # hc, mhc, mhclite,
                                        # geonorm, projspheretnorm, cayspheretnorm,
                                        # pspheretnorm
angle_upper_bound = 0.7853981633974483  # math.pi / 4
decay_method = "harmonic"               # harmonic, linear, or sqrt
init_method = "lns"                     # lns, or fixup
p = 3.0                                 # for p-SpheretNorm

# ----- I/O -----
seed = 1337
out_prefix_dataset = ""
out_prefix_model = ""
out_prefix_method = "residual"
out_dir = "out"
eval_interval = 200
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"      # 'scratch' or 'resume' or 'gpt2*'

# ----- wandb logging -----
wandb_log = True
wandb_project = "owt"
wandb_run_name = "exp"
wandb_group = "default"
wandb_notes = ""

# ----- data -----
dataset = "openwebtext"
gradient_accumulation_steps = 32
batch_size = 12
block_size = 1024

# ----- model -----
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True
scale_residual_init = True

# ----- AdamW optimizer -----
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# ----- learning rate schedule -----
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# ----- DDP -----
backend = "nccl"

# ----- system -----
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
