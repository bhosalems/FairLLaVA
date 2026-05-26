from typing import Optional, Dict
import os

import torch
import numpy as np
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
from open_clip.factory import HF_HUB_PREFIX, _MODEL_CONFIGS, load_state_dict



def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def from_pretrained(
        model_name: str,
        config: Optional[Dict] = None,
        checkpoint_path: str = None
    ):
    if (not model_name.startswith(HF_HUB_PREFIX)
        and model_name not in _MODEL_CONFIGS
        and config is not None):
        _MODEL_CONFIGS[model_name] = config
    # checkpoint_path = "/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/llava-rad_hf/biomedclipcxr_518_checkpoint.pt"
    print("===============checkpoint_path:=============\n", checkpoint_path)
    model, preprocess = create_model_from_pretrained(
        model_name=model_name,
        pretrained=checkpoint_path
    )

    tokenizer = get_tokenizer(model_name)

    return model, preprocess, tokenizer


def remove_transformer_pooler_weights(
        checkpoint_path, new_path=None
    ):
    need_new = False
    state_dict = load_state_dict(checkpoint_path, weights_only=False)
    for key in list(state_dict.keys()):
        if key.startswith("text.transformer.pooler"):
            need_new = True
            state_dict.pop(key)
    if need_new:
        # Use process-specific temp file to avoid race conditions
        if new_path is None:
            import os
            import atexit
            import time
            
            # Only clean stale files if directory has many accumulated files (avoid I/O on every load!)
            temp_dir = "/tmp/biomed_clip1"
            if os.path.exists(temp_dir):
                try:
                    files = [f for f in os.listdir(temp_dir) if f.startswith("ckpt_") and f.endswith(".pt")]
                    # Only clean if > 50 files accumulated (once per many runs, not every time!)
                    if len(files) > 10:
                        current_time = time.time()
                        for filename in files:
                            filepath = os.path.join(temp_dir, filename)
                            # Remove files older than 1 hour (3600 seconds)
                            if os.path.isfile(filepath) and (current_time - os.path.getmtime(filepath)) > 3600:
                                try:
                                    os.remove(filepath)
                                except Exception:
                                    pass
                except Exception:
                    pass  # Ignore cleanup errors
            
            # Use PID for unique filename (works for DeepSpeed multi-GPU)
            # Each GPU process has its own PID, so no collisions
            pid = os.getpid()
            
            # Optional: Also include LOCAL_RANK for clarity in distributed training
            local_rank = os.environ.get('LOCAL_RANK', None)
            if local_rank is not None:
                new_path = f"{temp_dir}/ckpt_rank{local_rank}_pid{pid}.pt"
            else:
                new_path = f"{temp_dir}/ckpt_{pid}.pt"
            
            # Register cleanup function to delete temp file when process exits
            def cleanup_temp_checkpoint():
                try:
                    if os.path.exists(new_path):
                        os.remove(new_path)
                        # Try to remove directory if empty
                        try:
                            os.rmdir(os.path.dirname(new_path))
                        except OSError:
                            pass  # Directory not empty or already removed
                except Exception:
                    pass  # Ignore cleanup errors
            
            atexit.register(cleanup_temp_checkpoint)
        
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        torch.save(state_dict, new_path)
        return new_path
    return checkpoint_path


if __name__ == "__main__":
    import sys
    remove_transformer_pooler_weights(*sys.argv[1:])