"""QLoRA fine-tuning of Qwen2.5-3B-Instruct for OPERA Cloud / OHIP grounding (Phase 2).

Target hardware: a single NVIDIA T4 (16 GB, Turing). Everything here is chosen to
fit that card:

  * 4-bit NF4 base weights (bitsandbytes)  -> ~2 GB for a 3B model
  * fp16 compute, NOT bf16                 -> Turing has no bf16 support
  * gradient checkpointing + batch size 1  -> activations at 4k context stay small
  * paged AdamW                            -> survives optimiser memory spikes
  * LoRA rank 32 on q/k/v/o_proj only      -> ~30 M trainable params (~1% of model)

Full-parameter fine-tuning of even the 3B is INFEASIBLE here (weights + grads +
Adam states ≈ 24 GB > 16 GB). That is a scope boundary of this dissertation, not
an oversight.

Usage on the Lightning.ai T4 studio:

    pip install -r requirements-train.txt
    python src/train_qlora.py                       # train
    python src/train_qlora.py --merge               # merge adapter -> fp16 model
    # then convert to GGUF with llama.cpp (see --merge output for exact commands)

Outputs:
    models/qwen2.5-3b-lora-adapter/     LoRA adapter (small, committable)
    models/qwen2.5-3b-lora-merged/      merged fp16 model (large, gitignored)
"""

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = MODELS_DIR / "qwen2.5-3b-lora-adapter"
MERGED_DIR = MODELS_DIR / "qwen2.5-3b-lora-merged"

TRAIN_FILE = DATA_DIR / "sft_train.jsonl"
VAL_FILE = DATA_DIR / "sft_val.jsonl"

# Measured with the Qwen2.5 tokenizer over the built dataset: p95 = 3,594 tokens,
# max = 4,416. 4,608 keeps every example intact — important because truncation
# would cut the *answer* off the end and leave a zero-loss example.
MAX_SEQ_LENGTH = 4608
SEED = 42


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true",
                   help="Merge a trained adapter into the base model and exit.")
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--output-dir", default=str(ADAPTER_DIR))
    p.add_argument("--merged-dir", default=str(MERGED_DIR))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap training steps (smoke-testing the GPU path cheaply).")
    p.add_argument("--no-liger", dest="liger", action="store_false",
                   help="Disable the fused Liger loss (needs more VRAM).")
    p.set_defaults(liger=True)
    return p.parse_args()


# --------------------------------------------------------------------------- #
def load_datasets():
    from datasets import Dataset

    def read(path):
        """Emit TRL *prompt-completion* (conversational) format, not `messages`.

        This matters: `completion_only_loss=True` is supported only for
        prompt-completion datasets. Handed a plain `messages` dataset, TRL falls
        back to computing loss over the whole sequence — and since the retrieved
        context is ~50x longer than the answer (median 2,210 vs 44 tokens), the
        model would mostly learn to reproduce context instead of answering from it.
        Splitting into prompt=[system,user] / completion=[assistant] makes TRL
        build a completion_mask so the loss lands on the answer only.
        """
        rows = []
        with open(path) as f:
            for line in f:
                m = json.loads(line)["messages"]
                rows.append({"prompt": m[:2], "completion": m[2:]})
        return Dataset.from_list(rows)

    train = read(TRAIN_FILE)
    val = read(VAL_FILE)
    print(f"train: {len(train)} examples | val: {len(val)} examples")
    return train, val


def train(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device. This script is meant for the Lightning.ai T4 — "
            "it will not run on the MacBook."
        )
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({vram:.1f} GB)")

    # Pick the compute dtype from the *hardware* capability, not from
    # torch.cuda.is_bf16_supported(): on torch >= 2.6 that helper counts software
    # emulation and returns True even on a T4 (capability 7.5, Turing), which has
    # no bf16 tensor cores. Trusting it would silently train through emulated
    # bf16 — far slower and numerically worse than plain fp16. Native bf16 starts
    # at Ampere (8.0).
    major, minor = torch.cuda.get_device_capability(0)
    supports_bf16 = major >= 8
    compute_dtype = torch.bfloat16 if supports_bf16 else torch.float16
    print(
        f"compute capability: {major}.{minor} | native bf16: {supports_bf16} "
        f"| compute dtype: {compute_dtype}"
    )

    train_ds, val_ds = load_datasets()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="sdpa",
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Attention projections only. MLP targets would roughly triple trainable
        # params for marginal gain on a grounding task, and cost T4 headroom.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        # Evaluation is what actually OOMs this card, not training. The default
        # eval batch size of 8 pads 8 long examples into one (8, ~3.8k) batch and
        # the loss upcasts logits to fp32 -> a single 17.3 GiB allocation on a
        # 14.5 GB card. Batch 1 + prediction_loss_only (which stops Trainer from
        # accumulating logits for every eval example) keeps it well inside VRAM.
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        fp16=not supports_bf16,
        bf16=supports_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_seq_length,
        # Fused linear+cross-entropy. Without it the loss materialises a
        # (1, seq, 151665) logits tensor and upcasts it to fp32 — at 4.4k tokens
        # that alone OOMs a 14.5 GB T4 (observed: a single 17.3 GiB allocation).
        # Liger computes the loss in chunks and never builds full logits.
        use_liger_kernel=args.liger,
        # Do NOT pack: packing concatenates unrelated examples across document
        # boundaries, which for a grounding objective would let one example's
        # answer sit next to another's context. Exactly the confusion we're curing.
        packing=False,
        # Train on the answer only (requires the prompt-completion dataset format
        # produced by load_datasets). Verified: TRL builds a completion_mask, and
        # only the assistant tokens contribute to the loss.
        completion_only_loss=True,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print("\n=== training ===")
    trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nadapter saved -> {args.output_dir}")

    metrics = trainer.state.log_history
    with open(Path(args.output_dir) / "train_log.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"training log -> {Path(args.output_dir) / 'train_log.json'}")
    print("\nNext: python src/train_qlora.py --merge")


# --------------------------------------------------------------------------- #
def merge(args):
    """Merge the LoRA adapter into fp16 base weights for GGUF conversion.

    Done on CPU in fp16: merging must not happen on the 4-bit quantised model
    (that would bake in quantisation error before we quantise again for GGUF).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"Loading base model in fp16 (CPU): {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    print(f"Applying adapter: {args.output_dir}")
    model = PeftModel.from_pretrained(base, args.output_dir)
    model = model.merge_and_unload()

    os.makedirs(args.merged_dir, exist_ok=True)
    model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.merged_dir)
    print(f"merged model -> {args.merged_dir}")

    print(f"""
=== Next: convert to GGUF Q4_K_M ===

  git clone https://github.com/ggerganov/llama.cpp
  pip install -r llama.cpp/requirements.txt
  python llama.cpp/convert_hf_to_gguf.py {args.merged_dir} \\
      --outfile {MODELS_DIR}/qwen2.5-3b-lora-f16.gguf --outtype f16
  cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build --target llama-quantize -j
  ./llama.cpp/build/bin/llama-quantize \\
      {MODELS_DIR}/qwen2.5-3b-lora-f16.gguf \\
      {MODELS_DIR}/qwen2.5-3b-lora-q4_k_m.gguf Q4_K_M

Then EVAL_CONFIGS['3B-LoRA'] already points at qwen2.5-3b-lora-q4_k_m.gguf.
""")


if __name__ == "__main__":
    args = parse_args()
    if args.merge:
        merge(args)
    else:
        train(args)
