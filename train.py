import argparse
import time
import torch
from transformers import AutoConfig, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--seq_length", type=int, default=4096)
    parser.add_argument("--global_batch_size", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    # distributed setup
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # load model config + weights
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if args.fp16:
        model = model.half()
    model.to(device)
    model.eval()

    # dummy input
    batch_size_per_gpu = args.global_batch_size // torch.distributed.get_world_size()
    vocab_size = config.vocab_size
    dummy_input = torch.randint(
        0, vocab_size,
        (batch_size_per_gpu, args.seq_length),
        dtype=torch.long, device=device
    )

    # warm-up
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy_input).logits

    # measure
    torch.cuda.synchronize(device)
    start = time.time()
    iters = 20
    for _ in range(iters):
        with torch.no_grad():
            _ = model(dummy_input).logits
    torch.cuda.synchronize(device)
    elapsed = time.time() - start

    # TFLOPs/s per GPU = (2 * ops per token) * batch * seq / elapsed / 1e12
    # ops per token ~ 6*N*H (approx), but we'll approximate with FLOPs from config
    params = sum(p.numel() for p in model.parameters())
    flops_per_token = 2 * params  # forward+backward roughly
    total_tokens = batch_size_per_gpu * args.seq_length * iters
    tflops_per_gpu = flops_per_token * total_tokens / elapsed / 1e12

    if local_rank == 0:
        print(f"Per-GPU throughput: {tflops_per_gpu:.2f} TFLOP/s")

if __name__ == "__main__":
    main()
