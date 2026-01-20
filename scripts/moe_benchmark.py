"""
Benchmark MoE (Mixture of Experts) model using megablocks.

From root directory of the project, run as:

python -m scripts.moe_benchmark

Example usage:
python -m scripts.moe_benchmark --hidden-size=2048 --ffn-hidden-size=8192 --num-experts=16 --top-k=2 --batch-size=16 --seq-len=512 --num-steps=100
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import time
import torch
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type

try:
    from megablocks import moe
    HAS_MEGABLOCKS = True
except ImportError:
    HAS_MEGABLOCKS = False
    print0("WARNING: megablocks not available. Install with: pip install megablocks or use ROCm extra")

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Benchmark MoE model")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--device-id", type=int, default=0, help="GPU device ID")
# Model architecture
parser.add_argument("--hidden-size", type=int, default=2048, help="hidden size of the model")
parser.add_argument("--ffn-hidden-size", type=int, default=8192, help="FFN hidden size")
parser.add_argument("--num-experts", type=int, default=16, help="total number of experts")
parser.add_argument("--top-k", type=int, default=2, help="number of active experts per token")
parser.add_argument("--expert-parallelism", type=int, default=1, help="expert parallelism (1 for single GPU)")
# Benchmark configuration
parser.add_argument("--batch-size", type=int, default=16, help="batch size")
parser.add_argument("--seq-len", type=int, default=512, help="sequence length")
parser.add_argument("--num-steps", type=int, default=100, help="number of training steps to benchmark")
parser.add_argument("--warmup-steps", type=int, default=10, help="number of warmup steps")
parser.add_argument("--bf16", action="store_true", default=True, help="use bfloat16 (default: True)")
parser.add_argument("--no-bf16", dest="bf16", action="store_false", help="disable bfloat16")
# Output
parser.add_argument("--verbose", action="store_true", help="print detailed timing information")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Setup device
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

if device_type == "cuda":
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)
    print0(f"Using device: {torch.cuda.get_device_name(args.device_id)}")
elif device_type == "mps":
    device = torch.device("mps")
    print0("Using device: MPS")
else:
    device = torch.device("cpu")
    print0("Using device: CPU")

autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if (device_type == "cuda" and args.bf16) else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Check megablocks availability
if not HAS_MEGABLOCKS:
    print0("ERROR: megablocks is required for this benchmark")
    print0("Install with: pip install megablocks or use the ROCm extra: uv pip install -e .[rocm]")
    exit(1)

# -----------------------------------------------------------------------------
# Configure the MoE Layer
print0("\n" + "="*80)
print0("MoE Benchmark Configuration")
print0("="*80)
print0(f"Hidden size: {args.hidden_size}")
print0(f"FFN hidden size: {args.ffn_hidden_size}")
print0(f"Number of experts: {args.num_experts}")
print0(f"Top-k (active experts per token): {args.top_k}")
print0(f"Expert parallelism: {args.expert_parallelism}")
print0(f"Batch size: {args.batch_size}")
print0(f"Sequence length: {args.seq_len}")
print0(f"Number of steps: {args.num_steps}")
print0(f"Warmup steps: {args.warmup_steps}")
print0(f"BF16: {args.bf16}")
print0("="*80 + "\n")

moe_config = moe.Config(
    hidden_size=args.hidden_size,
    ffn_hidden_size=args.ffn_hidden_size,
    num_experts=args.num_experts,
    top_k=args.top_k,
    device=device,
    bf16=args.bf16,
    expert_parallelism=args.expert_parallelism
)

# -----------------------------------------------------------------------------
# Initialize model
print0("Initializing MoE model...")
model = moe.dMoE(moe_config).to(device)
print0("Model initialized successfully")

# Calculate active parameters
active_params = args.hidden_size * args.ffn_hidden_size * args.top_k
total_params = sum(p.numel() for p in model.parameters())
print0(f"Total parameters: {total_params:,}")
print0(f"Active parameters per forward: {active_params:,}")

# -----------------------------------------------------------------------------
# Setup optimizer and dummy data
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dtype = torch.bfloat16 if args.bf16 else torch.float32
x = torch.randn(args.batch_size, args.seq_len, args.hidden_size, dtype=dtype, device=device)

# -----------------------------------------------------------------------------
# Benchmark
print0(f"\nStarting benchmark ({args.warmup_steps} warmup + {args.num_steps} benchmark steps)...")
print0("-"*80)

# Warmup
for step in range(args.warmup_steps):
    optimizer.zero_grad()
    with autocast_ctx:
        output, bias_loss = model(x)
        total_loss = torch.mean(output**2) + 0.1 * bias_loss
    total_loss.backward()
    optimizer.step()
    synchronize()

# Reset memory stats
if device_type == "cuda":
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

# Benchmark
forward_times = []
backward_times = []
step_times = []
total_times = []

for step in range(args.num_steps):
    step_start = time.time()
    
    optimizer.zero_grad()
    synchronize()
    
    forward_start = time.time()
    with autocast_ctx:
        output, bias_loss = model(x)
        total_loss = torch.mean(output**2) + 0.1 * bias_loss
    synchronize()
    forward_end = time.time()
    
    backward_start = time.time()
    total_loss.backward()
    synchronize()
    backward_end = time.time()
    
    optimizer.step()
    synchronize()
    step_end = time.time()
    
    forward_time = forward_end - forward_start
    backward_time = backward_end - backward_start
    step_time = step_end - step_start
    
    forward_times.append(forward_time)
    backward_times.append(backward_time)
    step_times.append(step_time)
    total_times.append(step_time)
    
    if args.verbose and (step + 1) % 10 == 0:
        print0(f"Step {step+1:4d}: forward={forward_time*1000:.2f}ms, backward={backward_time*1000:.2f}ms, total={step_time*1000:.2f}ms")

# -----------------------------------------------------------------------------
# Results
print0("-"*80)
print0("\nBenchmark Results")
print0("="*80)

# Calculate statistics
def stats(times):
    times = sorted(times)
    return {
        'mean': sum(times) / len(times),
        'median': times[len(times) // 2],
        'min': min(times),
        'max': max(times),
        'p95': times[int(len(times) * 0.95)],
        'p99': times[int(len(times) * 0.99)] if len(times) > 1 else times[0]
    }

forward_stats = stats(forward_times)
backward_stats = stats(backward_times)
step_stats = stats(step_times)

# Calculate throughput
tokens_per_step = args.batch_size * args.seq_len
forward_throughput = tokens_per_step / forward_stats['mean']
backward_throughput = tokens_per_step / backward_stats['mean']
total_throughput = tokens_per_step / step_stats['mean']

# Memory usage
peak_memory = get_max_memory() / (1024**3)  # GB

print0(f"\nForward Pass:")
print0(f"  Mean:   {forward_stats['mean']*1000:8.2f} ms")
print0(f"  Median: {forward_stats['median']*1000:8.2f} ms")
print0(f"  Min:    {forward_stats['min']*1000:8.2f} ms")
print0(f"  Max:    {forward_stats['max']*1000:8.2f} ms")
print0(f"  P95:    {forward_stats['p95']*1000:8.2f} ms")
print0(f"  P99:    {forward_stats['p99']*1000:8.2f} ms")
print0(f"  Throughput: {forward_throughput:,.0f} tokens/sec")

print0(f"\nBackward Pass:")
print0(f"  Mean:   {backward_stats['mean']*1000:8.2f} ms")
print0(f"  Median: {backward_stats['median']*1000:8.2f} ms")
print0(f"  Min:    {backward_stats['min']*1000:8.2f} ms")
print0(f"  Max:    {backward_stats['max']*1000:8.2f} ms")
print0(f"  P95:    {backward_stats['p95']*1000:8.2f} ms")
print0(f"  P99:    {backward_stats['p99']*1000:8.2f} ms")
print0(f"  Throughput: {backward_throughput:,.0f} tokens/sec")

print0(f"\nTotal Step (forward + backward + optimizer):")
print0(f"  Mean:   {step_stats['mean']*1000:8.2f} ms")
print0(f"  Median: {step_stats['median']*1000:8.2f} ms")
print0(f"  Min:    {step_stats['min']*1000:8.2f} ms")
print0(f"  Max:    {step_stats['max']*1000:8.2f} ms")
print0(f"  P95:    {step_stats['p95']*1000:8.2f} ms")
print0(f"  P99:    {step_stats['p99']*1000:8.2f} ms")
print0(f"  Throughput: {total_throughput:,.0f} tokens/sec")

if device_type == "cuda":
    print0(f"\nPeak GPU Memory: {peak_memory:.2f} GB")

print0(f"\nActive parameters per forward: {active_params:,}")
print0(f"Total model parameters: {total_params:,}")
print0("="*80)

# Cleanup
compute_cleanup()
