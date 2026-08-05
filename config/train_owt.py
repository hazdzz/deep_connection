eval_interval = 500
eval_iters = 200
log_interval = 10
always_save_checkpoint = False

wandb_log = False
wandb_project = 'deep_conn'
out_prefix_dataset = "owt"

dataset = 'openwebtext'
gradient_accumulation_steps = 8
batch_size = 16
block_size = 1024

dtype = 'bfloat16'
