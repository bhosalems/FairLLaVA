import argparse
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from transformers import GenerationConfig


def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map='cpu',
    )
    gc = getattr(model, "generation_config", None) or GenerationConfig()
    if not getattr(gc, "do_sample", False):  # Option A: deterministic default
        for k in ("temperature", "top_p", "top_k", "typical_p",
                "penalty_alpha", "epsilon_cutoff", "eta_cutoff"):
            if hasattr(gc, k):
                setattr(gc, k, None)
    # If you prefer sampling by default instead, do:
    # gc.do_sample = True

    model.generation_config = gc

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args)


# import os
# import torch
# from llava.model.builder import load_pretrained_model

# def main(src="microsoft/llava-rad",
#          base="lmsys/vicuna-7b-v1.5",
#          out_dir="./checkpoints/llavarad-merged",
#          projector_path="./checkpoints/llavarad_merged/mm_projector.bin",
#          dtype="fp16",
#          device="cuda"):
#     os.makedirs(out_dir, exist_ok=True)
#     torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

#     # Load exactly like eval: this merges LoRA into base and returns a plain LLaVA model
#     tokenizer, model, _, _ = load_pretrained_model(
#         model_path=src,
#         model_base=base,
#         model_name="llavarad",   # triggers the LoRA+merge path in your builder
#         load_8bit=False,
#         load_4bit=False,
#         device_map="auto",
#         device=device,
#     )
#     gc = getattr(model, "generation_config", None)
#     if gc is not None:
#         try:
#             # If not sampling, reset sampling-only params to safe defaults
#             if not getattr(gc, "do_sample", False):
#                 if hasattr(gc, "temperature"):
#                     gc.temperature = 1.0   # or set to None if your HF version allows
#                 if hasattr(gc, "top_p"):
#                     gc.top_p = 1.0         # or set to None if your HF version allows
#             model.generation_config = gc
#         except Exception as e:
#             print(f"[WARN] Skipping generation_config sanitation: {e}")

#     model.save_pretrained(out_dir)
#     tokenizer.save_pretrained(out_dir)
#     print(f"Saved merged model to: {out_dir}")

#     # Export projector weights in the expected format
#     state = {}
#     mm_proj = model.get_model().mm_projector.state_dict()
#     for k, v in mm_proj.items():
#         state[f"mm_projector.{k}"] = v.to(dtype=torch_dtype, device="cpu")

#     # Optional: include embed_tokens for the mm_use_im_start_end path
#     try:
#         state["model.embed_tokens.weight"] = (
#             model.get_input_embeddings().weight.detach().to(dtype=torch_dtype, device="cpu")
#         )
#         print("Included model.embed_tokens.weight in projector file.")
#     except Exception:
#         print("Skipped embed_tokens export (not required unless mm_use_im_start_end=True).")

#     # Ensure projector directory exists (note: your default path uses an underscore)
#     os.makedirs(os.path.dirname(projector_path), exist_ok=True)
#     torch.save(state, projector_path)
#     print(f"Exported projector to: {projector_path}")

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--src", default="microsoft/llava-rad")
#     ap.add_argument("--base", default="lmsys/vicuna-7b-v1.5")
#     ap.add_argument("--out_dir", default="./checkpoints/llavarad-merged")
#     ap.add_argument("--projector_path", default="./checkpoints/llavarad_merged/mm_projector.bin")
#     ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
#     ap.add_argument("--device", default="cuda")
#     args = ap.parse_args()
#     main(args.src, args.base, args.out_dir, args.projector_path, args.dtype, args.device)
