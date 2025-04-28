#!/usr/bin/env python3
import argparse
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    # swallow DeepSpeed’s local_rank flag
    parser.add_argument("--local_rank", type=int, default=0, help="(unused) for deepspeed launcher")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained LLaMA model"
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=4096,
        help="Sequence length for training/inference"
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=1024,
        help="Total batch size across all GPUs"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed-precision"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # set device
    device = torch.device(f"cuda:{args.local_rank}")

    # load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if args.fp16:
        model = model.half()
    model.to(device)
    model.eval()

    # prepare dummy input
    batch_size_per_gpu = args.global_batch_size // torch.cuda.device_count()
    dummy_input = torch.randint(
        low=0,
        high=tokenizer.vocab_size,
        size=(batch_size_per_gpu, args.seq_length),
        device=device,
        dtype=torch.long
    )

    # warm-up
    for _ in range(5):
        _ = model(dummy_input)

    # timing
    torch.cuda.synchronize()
    start = time.time()
    outputs = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # compute TFLOPS: roughly 2 * F * B * S / T
    # F = number of parameters, B = batch, S = seq length, T = elapsed
    num_params = sum(p.numel() for p in model.parameters())
    tflops = 2 * num_params * batch_size_per_gpu * args.seq_length / (elapsed * 1e12)

    if args.local_rank == 0:
        print(f"Per-GPU throughput: {tflops:.1f} TFLOP/s on GPU {device}")

if __name__ == "__main__":
    main()
