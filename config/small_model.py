wandb_group = "small"
out_prefix_model = "small"

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

learning_rate = 6e-4
min_lr = 6e-5
max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
