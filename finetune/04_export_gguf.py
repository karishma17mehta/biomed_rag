"""
finetune/04_export_gguf.py

Merge the LoRA adapter into the base model and export to GGUF format
so the fine-tuned model can be served locally via Ollama.

Steps performed:
  1. Load base model + LoRA adapter
  2. Merge adapter weights into base (produces a full fp16 model)
  3. Save merged model to finetune/merged-model/
  4. Export to GGUF (Q4_K_M quantisation) for Ollama
  5. Print the Modelfile and ollama commands to run the model

Usage:
    python finetune/04_export_gguf.py --adapter finetune/lora-adapter

Outputs:
    finetune/merged-model/         full fp16 merged weights (large, ~14 GB)
    finetune/gguf/model-q4_k_m.gguf  4-bit quantised GGUF (~4 GB)
    finetune/Modelfile             ready-to-use Ollama Modelfile
"""

import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="finetune/lora-adapter")
    p.add_argument("--base_model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    p.add_argument("--merged_dir", default="finetune/merged-model")
    p.add_argument("--gguf_dir", default="finetune/gguf")
    p.add_argument(
        "--quantisation",
        default="q4_k_m",
        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
        help="GGUF quantisation level (q4_k_m = best size/quality trade-off)",
    )
    p.add_argument(
        "--hf_repo",
        default=None,
        help="Optional HuggingFace repo ID to push merged model (e.g. your-org/pubmedqa-mistral-7b)",
    )
    return p.parse_args()


def merge_and_save(base_model: str, adapter: str, merged_dir: str) -> None:
    from unsloth import FastLanguageModel

    print(f"Loading base model + adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)

    print("Merging adapter into base weights (this takes a few minutes)...")
    model = model.merge_and_unload()

    out = Path(merged_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"Merged model saved → {out}")


def export_gguf(base_model: str, adapter: str, gguf_dir: str, quantisation: str) -> Path:
    """Use unsloth's built-in GGUF exporter (wraps llama.cpp convert)."""
    from unsloth import FastLanguageModel

    print(f"\nExporting to GGUF ({quantisation})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter)

    gguf_path = Path(gguf_dir)
    gguf_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained_gguf(
        str(gguf_path / "model"),
        tokenizer,
        quantization_method=quantisation,
    )

    # unsloth names the file model-{quantisation}.gguf
    expected = gguf_path / f"model-{quantisation}.gguf"
    print(f"GGUF file → {expected}")
    return expected


def write_modelfile(gguf_path: Path, modelfile_path: Path) -> None:
    content = f"""FROM {gguf_path.resolve()}

SYSTEM \"\"\"You are a biomedical research assistant trained on PubMedQA.
Given a research question and the body of a PubMed abstract (the conclusion is withheld),
answer with exactly one word — 'yes', 'no', or 'maybe' — then write the conclusion.

Format your response as:
Answer: <yes|no|maybe>

Conclusion: <your conclusion>\"\"\"

PARAMETER temperature 0.1
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
"""
    modelfile_path.write_text(content)
    print(f"Modelfile written → {modelfile_path}")


def push_to_hub(merged_dir: str, hf_repo: str) -> None:
    from huggingface_hub import HfApi

    print(f"\nPushing merged model to HuggingFace Hub: {hf_repo}")
    api = HfApi()
    api.upload_folder(
        folder_path=merged_dir,
        repo_id=hf_repo,
        repo_type="model",
    )
    print(f"  → https://huggingface.co/{hf_repo}")


def main():
    args = parse_args()

    # Step 1 — export to GGUF (includes merge internally via unsloth)
    gguf_file = export_gguf(args.base_model, args.adapter, args.gguf_dir, args.quantisation)

    # Step 2 — write Modelfile
    modelfile_path = Path("finetune") / "Modelfile"
    write_modelfile(gguf_file, modelfile_path)

    # Step 3 — optionally push merged fp16 model to HF Hub
    if args.hf_repo:
        merge_and_save(args.base_model, args.adapter, args.merged_dir)
        push_to_hub(args.merged_dir, args.hf_repo)

    print("\n" + "=" * 60)
    print("Deployment instructions")
    print("=" * 60)
    print("\n[Local — Ollama]")
    print(f"  ollama create pubmedqa-mistral -f {modelfile_path}")
    print("  ollama run pubmedqa-mistral")
    print()
    print("[API server — vllm]  (requires merged model)")
    print(f"  vllm serve {args.merged_dir} --port 8000 --dtype bfloat16")
    print()
    print("[HuggingFace Inference Endpoint]")
    print("  Upload merged model to HF Hub, then deploy via:")
    print("  https://ui.endpoints.huggingface.co/new")


if __name__ == "__main__":
    main()
