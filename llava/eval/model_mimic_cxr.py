"""
A model worker executes the model.
"""
import os
import json
import math
from pathlib import Path
import torch
import fire
from tqdm import tqdm
from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import build_logger, disable_torch_init, data_loaders
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from scripts.dac_infereval import load_dac_checkpoint

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def create_batches(data, batch_size, group_by_length, tokenizer):
    if batch_size == 1 or not group_by_length:
        return [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    else:
        batches = []
        batch, batch_len = [], None
        for d in data:
            d_len = len(tokenizer(d["conversations"][0]['value']).input_ids)
            if batch_len is None or d_len == batch_len:
                batch_len = d_len
                batch.append(d)
                if len(batch) == batch_size:
                    batches.append(batch)
                    batch, batch_len = [], None
            else:
                assert len(batch)
                batches.append(batch)
                batch, batch_len = [d], d_len
        if len(batch):
            batches.append(batch)
        assert len(data) == sum(len(b) for b in batches)
        return batches


def eval_model(
        query_file: str,
        image_folder: str,
        conv_mode: str,
        prediction_file: str,
        model_path: str,
        model_base: str = None,
        load_8bit: bool = False,
        load_4bit: bool = False,
        device: str = "cuda",
        temperature: float = 0.2,
        top_p: float = None,
        num_beams: int = 1,
        chunk_idx: int = 0,
        num_chunks: int = 1,
        batch_size: int = 8,
        loader: str = "default",
        group_by_length: bool = False,
        fairness_finetune: bool = False,
    ):
    os.makedirs("logs", exist_ok=True)
    logger = build_logger("model_mimic_cxr", f"logs/model_mimic_cxr_{chunk_idx}.log")


    print("[CHECK] model_path:", model_path)
    print("[CHECK] model_base:", model_base)

    # load model
    disable_torch_init()
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path("microsoft/llava-rad") # TODO we need to change this, instead of using hard code, but works for now as long as it is LLava style model.
    # Doesnt take names other than LLavarad."
    if not model_name.startswith("llavarad"):
        # "llava" needs to be in model_name to correctly load the model.
        raise ValueError(f"Model name {model_name} is not 'llavarad'.")
    logger.info(f"Loading the model {model_name} ...")
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device)
    model.fairness_finetune = fairness_finetune # disable fairness finetune

    # Resolve the actual device the model ended up on.
    # Some loading paths may place the model on CPU (e.g., offload/device_map).
    # Never assume CUDA here; instead, move inputs/images to model_device.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device if device else "cpu")
    # load_dac_checkpoint(model, "/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-20e-1e-4-20251005153413/checkpoint-3/dac_modules.bin")
    # model.fairness_finetune = False  # no fairness head in eval, explicitly set to False
    try:
        from peft import PeftModel
        is_peft = isinstance(model, PeftModel)
    except Exception:
        is_peft = hasattr(model, "peft_config")
    print("[CHECK] is PEFT/LoRA model:", is_peft)
    if is_peft:
        active = getattr(model, "active_adapter", "default")
        print("[CHECK] active_adapter:", active)
        lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n]
        print("[CHECK] #LoRA tensors:", len(lora_params), "total LoRA params:",
            sum(p.numel() for _, p in lora_params))
        if lora_params:
            with torch.no_grad():
                m = torch.stack([p.float().abs().mean() for _, p in lora_params]).mean().item()
            print("[CHECK] mean(|LoRA|) =", m)

    proj_file = Path(model_path) / "mm_projector.bin"
    has_proj_file = proj_file.exists()
    print("[CHECK] mm_projector.bin present:", has_proj_file)
    if has_proj_file and hasattr(model, "get_model") and hasattr(model.get_model(), "mm_projector"): # This is a denbug and can be removed?
        disk = torch.load(proj_file, map_location="cpu")
        # Map keys: your saved file likely used full names. Try several match strategies.
        model_sd = dict(model.named_parameters())
        # find any overlapping projector key
        match_key = None
        for k in disk.keys():
            for cand in (k, f"model.{k}"):
                if cand in model_sd:
                    match_key = (k, cand)
                    break
            if match_key: break
        print("[CHECK] projector key match found:", match_key is not None, match_key)
        if match_key:
            k_disk, k_model = match_key
            same = torch.allclose(model_sd[k_model].detach().cpu(), disk[k_disk], atol=1e-6)
            print("[CHECK] projector tensor equality (sample key):", same)
    else:
        print("[WARN] projector not found on disk or not attached on model; "
            "if you meant to use it, double-check save path and loading logic.")
    
    # load data
    from types import SimpleNamespace
    data_args = SimpleNamespace(data_path=query_file, fairness_labels=False)
    all_queries = data_loaders[loader](data_args)
    if group_by_length:
        all_queries = sorted(all_queries, key=lambda x: len(tokenizer(x["conversations"][0]['value']).input_ids))
    queries = get_chunk(all_queries, num_chunks, chunk_idx)
    logger.info(f"Loaded {len(queries)} / {len(all_queries)} ({chunk_idx}:{num_chunks}) examples.")

    os.makedirs(os.path.dirname(prediction_file), exist_ok=True)
    pred_file = open(prediction_file, "w")
    log_prediction = True
    batches = create_batches(queries, batch_size, group_by_length, tokenizer)
    for batch_queries in tqdm(batches):
        batch_prompts = []
        batch_input_ids = []
        batch_images = []
        for query in batch_queries:
            q = query["conversations"][0]["value"]

            q = q.replace("<image>", "").strip()
            if model.config.mm_use_im_start_end:
                q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
            else:
                q = DEFAULT_IMAGE_TOKEN + '\n' + q

            conv= conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

            image = Image.open(os.path.join(image_folder, query["image"])).convert("RGB")
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            batch_prompts.append(prompt)
            batch_input_ids.append(input_ids)
            batch_images.append(image_tensor)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        

        with torch.inference_mode():
            input_ids_t = torch.stack(batch_input_ids).to(model_device)
            images_t = torch.stack(batch_images)
            if model_device.type == "cuda":
                images_t = images_t.half()
            images_t = images_t.to(model_device)
            batch_output_ids = model.generate(
                input_ids_t,
                images=images_t,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=256,
                use_cache=True).cpu()

            batch_outputs = tokenizer.batch_decode(
                batch_output_ids[:, len(batch_input_ids[0]):], skip_special_tokens=True
            )

        for query, prompt, outputs, input_ids, output_ids in zip(
            batch_queries, batch_prompts, batch_outputs, batch_input_ids, batch_output_ids):
            q = query["conversations"][0]["value"]
            ref = query["conversations"][1]["value"]
            input_token_len = input_ids.shape[0]
            n_diff_input_output = (input_ids != output_ids[:input_token_len]).sum().item()
            if n_diff_input_output > 0:
                logger.warning(f'{n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            # Be permissive about ID fields across datasets.
            current_id = (
                query.get("id")
                or query.get("ImageID")
                or query.get("image_id")
                or query.get("imageId")
                or query.get("image")
            )
            if current_id is None:
                raise KeyError("Could not determine sample id from query (expected one of: id, ImageID, image_id, image)")
            
            if log_prediction:
                logger.info(f"id: {current_id}")
                logger.info(f"query: {repr(q)}")
                logger.info(f"prompt: {repr(prompt)}")
                logger.info(f"reference: {repr(ref)}")
                logger.info(f"prediction: {repr(outputs)}")

            # Include optional label + demographics when present (useful for fairness eval).
            extra = {}
            for k in ("label", "Age", "SexEncoded", "gender", "anchor_age_group", "race_major", "image"):
                if k in query:
                    extra[k] = query[k]
            pred_file.write(
                json.dumps(
                    {"id": current_id, "query": q, "reference": ref, "prediction": outputs, **extra},
                    ensure_ascii=False,
                )
                + "\n"
            )
            pred_file.flush()
        
        log_prediction = False

    pred_file.close()


if __name__ == "__main__":
    fire.Fire(eval_model)
