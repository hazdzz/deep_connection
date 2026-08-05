import os
import numpy as np
import time
import torch

from contextlib import nullcontext
from model.gpt_base import GPTConfig, GPT

# -----------------------------------------------------------------------------
batch_size = 12
block_size = 1024
bias = False
real_data = True
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = True   # use PyTorch 2.0 to compile the model to be faster
profile = False  # use pytorch profiler, or just simple benchmarking?

# model architecture / deep-connection settings (overridable via configurator.py)
n_layer = 12
n_head = 12
n_embd = 768
deep_conn = "preln"                     # preln, postln, lns, periln, keel,
                                        # predyt, predyisru, prederf,
                                        # geonorm, projspheretnorm, cayspheretnorm,
                                        # pspheretnorm
beta_upper_bound = 0.7853981633974483   # math.pi / 4, used by geonorm
decay_method = "harmonic"               # harmonic, linear, or sqrt (geonorm)
init_method = "rezero"                  # rezero, lns, or fixup (sphere norms)
p = 3.0                                 # for p-SpheretNorm

exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True         # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# data loading init
if real_data:
    dataset = 'openwebtext'
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    def get_batch(split):
        data = train_data  # note: ignore split in benchmarking script
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x, y
else:
    # alternatively, if fixed data is desired to not care about data loading
    x = torch.randint(50304, (batch_size, block_size), device=device)
    y = torch.randint(50304, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

# model init (pass the deep-connection args so we can benchmark each method)
gptconf = GPTConfig(
    block_size=block_size,             # how far back does the model look? i.e. context size
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,  # size of the model
    dropout=0,                         # for determinism
    bias=bias,
    deep_conn=deep_conn,
    beta_upper_bound=beta_upper_bound,
    decay_method=decay_method,
    init_method=init_method,
    p=p,
)
model = GPT(gptconf)
model.to(device)

optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

if compile:
    print("Compiling model...")
    model = torch.compile(model)  # pytorch 2.0

if profile:
    # useful docs on pytorch profiler:
    # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,    # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step()  # notify the profiler at end of each step

else:

    # simple benchmarking
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
        # reset peak-memory stats at the start of the measured stage
        if stage == 1 and device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        # mfu = model.estimate_mfu(batch_size * 1 * num_steps, dt)
        if stage == 1:
            # tokens_per_sec = batch_size * block_size * num_steps / dt
            # print(f"deep_conn: {deep_conn}")
            # print(f"time per iteration: {dt/num_steps*1000:.4f}ms, MFU: {mfu*100:.2f}%")
            # print(f"tokens/sec: {tokens_per_sec:.2f}")
            tokens_per_sec = batch_size * block_size * num_steps / dt
            print(f"deep_conn: {deep_conn}")
            print(f"time per iteration: {dt/num_steps*1000:.4f}ms")
            print(f"tokens/sec: {tokens_per_sec:.2f}")
            if device_type == 'cuda':
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                print(f"peak memory: {peak_gb:.2f} GB")
