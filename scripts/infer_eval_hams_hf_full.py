import os
import json
import subprocess
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import argparse

# Arguments for flexibility
parser = argparse.ArgumentParser(description="Run HuggingFace LLaVA inference and full HAMS evaluation pipeline.")
parser.add_argument('--model_id', type=str, default="YuchengShi/LLaVA-v1.5-7B-HAM10000")
parser.add_argument('--data_json', type=str, required=True)
parser.add_argument('--image_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--device', type=str, default="cuda")
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
out_jsonl = os.path.join(args.out_dir, "hf_hams_preds.jsonl")

# Load model and processor
model = LlavaForConditionalGeneration.from_pretrained(
    args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(args.device)
processor = AutoProcessor.from_pretrained(args.model_id)

# Load data
with open(args.data_json, "r") as f:
    data = json.load(f)

results = []
for entry in tqdm(data, desc="Running inference"):
    image_path = os.path.join(args.image_dir, entry["image"])
    # Use chat template for LLaVA prompt formatting
    # conversations = entry.get("conversations")
    # if conversations and isinstance(conversations, list) and len(conversations) > 0:
    #     # LLaVA expects a list of dicts with 'role' and 'content' keys
    #     # If your JSON already matches this, use as is; otherwise, convert
    #     conversation = conversations
    #     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    #     question = conversations[0].get("value", "")
    # else:
    #     # Fallback: use default question
    #     question = "<image>\nWhat type of skin lesion is this?"
    #     # Build a minimal conversation for the template
    #     conversation = [
    #         {"role": "user", "content": question}
    #     ]
    #     prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    raw_image = Image.open(image_path).convert("RGB")
    prompt="<image>\nWhat type of skin lesion is this?"
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(args.device, torch.float16)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    answer = processor.decode(output[0][2:], skip_special_tokens=True)
    # Store result
    # Extract short image id (e.g., ISIC_0026100 from path)
    image_path = entry["image"]
    short_id = os.path.splitext(os.path.basename(image_path))[0]
    results.append({
        "id": short_id,
        "query": prompt,
        "reference": entry.get("reference"),
        "prediction": answer,
        "label": entry.get("label"),
        "Age": entry.get("Age"),
        "SexEncoded": entry.get("SexEncoded"),
        "gender": entry.get("gender"),
        "anchor_age_group": entry.get("anchor_age_group"),
        "image": image_path,
        # Optionally keep demographics dict if present
        "demographics": entry.get("demographics"),
        # Optionally keep attribute if present
        "attribute": entry.get("attribute"),
    })

# Save predictions
with open(out_jsonl, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
print(f"Saved predictions to {out_jsonl}")

# Merge predictions (for compatibility with stratify_results.py)
merged_path = os.path.join(args.out_dir, "merged_preds.jsonl")
os.rename(out_jsonl, merged_path)

# Run stratification
print("Running stratification...")
subprocess.run([
    "python", "scripts/stratify_results.py",
    "--pred_dir", args.out_dir,
    "--dataset", "ham10000"
], check=True)

# Run fairness/summary evaluation
print("Running fairness/summary evaluation...")
subprocess.run([
    "python", "scripts/eval_ham_fairness.py",
    "--pred_dir", args.out_dir,
    "--query_file", args.data_json
], check=True)

print(f"All done. Results in {args.out_dir}")
