import time, torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--seq_length', type=int, default=4096)
    p.add_argument('--global_batch_size', type=int, default=1024)
    p.add_argument('--dp', type=int, default=1)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--pp', type=int, default=1)
    p.add_argument('--fp16', action='store_true')
    return p.parse_args()

def get_batch(tok, sl, bs, device):
    return torch.randint(0, tok.vocab_size, (bs, sl), device=device)

def main():
    args=parse_args()
    deepspeed.init_distributed()

    ds_cfg = {
      "train_micro_batch_size_per_gpu": args.global_batch_size//(args.dp*args.tp),
      "gradient_accumulation_steps": 1,
      "fp16": {"enabled": args.fp16},
      "zero_optimization": {"stage": 0}
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=(torch.float16 if args.fp16 else torch.float32))
    model, _, _, _ = deepspeed.initialize(model=model, config=ds_cfg)
    tok = AutoTokenizer.from_pretrained(args.model_path)

    batch = get_batch(tok, args.seq_length, ds_cfg["train_micro_batch_size_per_gpu"], model.local_rank)
    # Warm-up
    for _ in range(5):
        loss = model(batch, labels=batch).loss
        model.backward(loss); model.step()

    # Timed runs
    steps=20
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(steps):
        loss = model(batch, labels=batch).loss
        model.backward(loss); model.step()
    torch.cuda.synchronize(); t1=time.time()

    # Compute TFLOPs/GPU
    total_p = sum(p.numel() for p in model.module.parameters())
    flops = 2 * total_p * args.seq_length
    secs = (t1-t0)/steps
    tflops = flops / secs / 1e12
    print(f"=== TFLOP/s per GPU: {tflops:.0f} ===")

if __name__=='__main__':
    main()
EOF
