import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates


@torch.no_grad()
def load_dac_checkpoint(model, dac_path):
    sd = torch.load(dac_path, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if k.startswith("dac_modules.")}
    base = model.state_dict()
    mism = [(k, tuple(v.shape), tuple(base[k].shape))
            for k, v in sd.items() if k in base and v.shape != base[k].shape]
    missing = [k for k in sd.keys() if k not in base]
    unused  = [k for k in base.keys() if k.startswith("dac_modules.") and k not in sd]

    if mism:
        print("Shape mismatches:\n", "\n".join([f"{k}: {a} vs {b}" for k,a,b in mism]))
    if missing:
        print("Missing in base (but in checkpoint):\n", "\n".join(missing))
    if unused:
        print("Unused in checkpoint (base has extra):\n", "\n".join(unused))
    sd_ok = {k: v for k, v in sd.items() if k in base and v.shape == base[k].shape}
    msg = model.load_state_dict(sd_ok, strict=False)
    print("Loaded DAC with:", msg)

DEM_MAP = {
    "gender": {
        "male": 0, "m": 0,
        "female": 1, "f": 1
    },
    "race_major": {
        "white": 0,
        "black": 1, "black or african american": 1,
        "asian": 2,
        "hispanic": 3, "hispanic or latino": 3,
        "other": 4,
        "american indian or alaska native": 5,
        "native hawaiian or pacific islander": 6,
        "unknown": 7, "declined / unable to obtain": 7, "": 7,
    },
    # anchor_age_group is already numeric in training; include string fallbacks just in case
    "anchor_age_group": { "young": 0, "middle": 1, "old": 2 }
}

def encode_dem(values, attr_key, device):
    """
    Map a list of raw demographic values (strings/ints/Nones) into torch.long IDs
    using the exact same mapping as training.
    """
    m = DEM_MAP.get(attr_key, {})
    out = []
    for v in values:
        if v is None:
            key = ""
        elif isinstance(v, (int, float)):
            out.append(int(v))
            continue
        else:
            key = str(v).strip().lower()
        out.append(m.get(key, m.get("", 0)))
    return torch.tensor(out, dtype=torch.long, device=device)

@torch.no_grad()
def eval_dac(
    model,
    tokenizer,
    image_processor,
    data_path: str,     # json or jsonl (each line is a dict)
    image_folder: str,
    loader: str,        # unused now; kept for CLI compatibility
    attr_key: all,      # "gender" | "race_major" | "anchor_age_group"
    dac_bin: str,
    batch_size: int = 16,
    num_workers: int = 2,   # unused; kept for signature compat
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    load_dac_checkpoint(model, dac_bin)
    model.eval()    
    conv_mode = "v1"
    use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)

    def load_items(path):
        import json
        with open(path, "r") as f:
            first = f.read(1)
            f.seek(0)
            if first == "[":  # JSON list
                return json.load(f)
            # JSONL
            return [json.loads(line) for line in f if line.strip()]

    items = load_items(data_path)
    all_target_dems = ["gender", "race_major", "anchor_age_group"]    
    eval_dems = all_target_dems if attr_key == "all" else [attr_key] 
    
    def has_all_labels(ex):
        if "image" not in ex: return False
        if "conversations" not in ex or not ex["conversations"]: return False
        return all(k in ex for k in all_target_dems)

    items = [ex for ex in items if has_all_labels(ex)]
    if not items:
        print("[WARN] no valid examples with all required labels found.")
        return 0.0, None, None

    def batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    preds = {d: [] for d in eval_dems}       
    trues = {d: [] for d in eval_dems}
    logits_buf = {d: [] for d in eval_dems}
    num_batches = np.ceil(len(items) / batch_size)
    for chunk in tqdm(batches(items, batch_size), total=num_batches, desc="DAC eval", dynamic_ncols=True):
        # build inputs
        id_seqs = []
        imgs = []
        gt = {k: [] for k in all_target_dems}

        for ex in chunk:
            q = ex["conversations"][0]["value"].replace("<image>", "").strip()
            if use_im_start_end:
                q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + q
            else:
                q = DEFAULT_IMAGE_TOKEN + "\n" + q

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").squeeze(0)
            id_seqs.append(ids)

            img = Image.open(os.path.join(image_folder, "/".join(ex["image"].split("/")[1:]))).convert("RGB")
            px = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
            imgs.append(px)
            for dem in all_target_dems:
                gt[dem].append(ex[dem])

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = pad_sequence(id_seqs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = (input_ids != pad_id).long()
        imgs = torch.stack(imgs).to(device, dtype=next(model.parameters()).dtype)
        
        label_tensors = {dem: encode_dem(gt[dem], dem, device) for dem in all_target_dems}
        out = model(
            input_ids=input_ids,
            images=imgs,
            attention_mask=attention_mask,
            gender=label_tensors["gender"],
            race_major=label_tensors["race_major"],
            anchor_age_group=label_tensors["anchor_age_group"],
            return_dict=True,
            dac_evaluation=True,
        )

        for dem in eval_dems:                         
            if dem == "race_major":
                i_dem = "race"
            elif dem == "anchor_age_group":
                i_dem = "age"
            else:
                i_dem = dem
            d = out[i_dem]
            preds[dem].append(d["pred"].detach().cpu())
            trues[dem].append(d["label"].detach().cpu())
            if "logits" in d and d["logits"] is not None:
                logits_buf[dem].append(d["logits"].detach().cpu())
    
    for dem in eval_dems:                            
        y_pred = torch.cat(preds[dem])
        y_true = torch.cat(trues[dem])

        acc = (y_pred == y_true).float().mean().item()
        K = int(max(1, y_true.max().item() + 1))
        cm = torch.zeros(K, K, dtype=torch.long)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        print(f"\n== DAC ({dem}) accuracy: {acc*100:.2f}% (N={len(y_true)})")
        print("Confusion matrix (rows=true, cols=pred):")
        for r in range(K):
            print(" ".join(f"{int(cm[r,c]):5d}" for c in range(K)))

        per_class = []
        for k in range(K):
            denom = cm[k].sum().item()
            per_class.append((cm[k, k].item() / max(1, denom)) if denom > 0 else 0.0)
        print("Per-class accuracies:", [f"{100*x:.1f}%" for x in per_class])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_base", default=None)
    ap.add_argument("--model_name", default="llavarad")
    ap.add_argument("--data_path", required=True, help="Same JSON you trained on")
    ap.add_argument("--image_folder", required=True)
    ap.add_argument("--loader", default="default", help="data_args.loader string used in training")
    ap.add_argument("--attr_key", required=True, choices=["all", "gender","race_major","anchor_age_group"])
    ap.add_argument("--dac_bin", default="", help="path to dac_modules.bin (defaults to model_path/dac_modules.bin if omitted)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, args.model_name, 
        load_8bit=False, load_4bit=False, device="cuda:0",
    )
    model.to("cuda:0").eval()
    
    dac_bin = args.dac_bin or os.path.join(args.model_path, "dac_modules.bin")
    eval_dac(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        data_path=args.data_path,
        image_folder=args.image_folder,
        loader=args.loader,
        attr_key=args.attr_key,
        dac_bin=dac_bin,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

if __name__ == "__main__":
    main()
