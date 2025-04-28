#!/usr/bin/env python3
import os
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Measure TFLOP/s per GPU")
    parser.add_argument("--model_path", required=True, help="Path to a HF llama-3-8b or llama-3-70b folder")
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--global_batch_size", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    # allow DeepSpeed’s local_rank
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return parser.parse_args()

def main():
    # initialize distributed if launched under deepspeed or torch.distributed
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda")

    # Load model only (no tokenizer needed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        low_cpu_mem_usage=True,
        device_map={"": args.local_rank}
    )
    model.eval()

    # prepare dummy inputs
    per_gpu_batch = args.global_batch_size // torch.distributed.get_world_size()
    vocab_size = model.config.vocab_size
    input_ids = torch.randint(0, vocab_size,
                              (per_gpu_batch, args.seq_length),
                              device=device, dtype=torch.long)

    # count params
    total_params = sum(p.numel() for p in model.parameters())
    # FLOPs per token step ~ 2 × params
    flops_per_token = 2 * total_params

    # warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids).logits

    # benchmark
    iters = 20
    torch.distributed.barrier()
    start = time.time()
    for _ in range(iters):
        with torch.no_grad():
            _ = model(input_ids).logits
    torch.distributed.barrier()
    elapsed = time.time() - start

    steps_per_sec = iters / elapsed
    # each forward is ~ flops_per_token × seq_length × batch_per_gpu
    tflops_per_sec_per_gpu = (flops_per_token * args.seq_length * per_gpu_batch * steps_per_sec) / 1e12

    if args.local_rank == 0:
        print(f"\n=== Benchmark results ===")
        print(f"Model:          {args.model_path}")
        print(f"Total params:   {total_params/1e9:.2f} B")
        print(f"seq_length:     {args.seq_length}")
        print(f"global batch:   {args.global_batch_size}  (per GPU: {per_gpu_batch})")
        print(f"Steps/sec/GPU:  {steps_per_sec:.2f}")
        print(f"TFLOP/s/GPU:    {tflops_per_sec_per_gpu:.1f}\n")

if __name__ == "__main__":
    # initialize distributed if launched under deepspeed or torch.distributed
    #torch.distributed.init_process_group(backend="nccl", init_method="env://")
    main()

