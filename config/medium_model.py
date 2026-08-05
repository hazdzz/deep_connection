wandb_group = "medium"
out_prefix_model = "medium"

n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.0
bias = True

learning_rate = 3e-4
min_lr = 3e-5
max_iters = 50000
lr_decay_iters = 50000
warmup_iters = 2000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
