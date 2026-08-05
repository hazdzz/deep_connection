wandb_group = "large"
out_prefix_model = "large"

n_layer = 36
n_head = 20
n_embd = 1280
dropout = 0.0
bias = True

learning_rate = 2.5e-4
min_lr = 2.5e-5
max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

