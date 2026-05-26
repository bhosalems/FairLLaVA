import os
import json
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers.modeling_utils import unwrap_model
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, data_loaders
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from jiwer import wer, cer, mer, wil


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# NOTE: This script evaluates handwriting transcription (HTR) performance for a set of (image, text) pairs.
# Expected input format: JSON or JSONL.
# Each example must contain fields: {"image": <path or relative path>, "text": <ground truth transcription>, "conversations": [...] optional for prompt style}
# If conversations absent, we build a default prompt with just the image token and a query like "Please transcribe the handwriting.".
# Output: metrics printed + per-sample results saved optionally.

@dataclass
class HTRExample:
    image: str
    text: str
    conversations: Any = None
    # Optional demographic attributes if fairness breakdown desired
    gender: str = None
    race_major: str = None
    anchor_age_group: str = None


def load_items(path: str) -> List[Dict[str, Any]]:
    """Generic loader for JSON or JSONL."""
    with open(path, 'r') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def build_prompt(example: Dict[str, Any], use_im_start_end: bool, conv_mode: str = 'v1') -> str:
    # Reuse conversation template logic if conversations provided, else create default.
    if 'conversations' in example and example['conversations']:
        # Assume first user turn contains <image> token placeholder or we insert it.
        q = example['conversations'][0]['value']
        if '<image>' in q:
            q = q.replace('<image>', '').strip()
        if use_im_start_end:
            q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + q
        else:
            q = DEFAULT_IMAGE_TOKEN + '\n' + q
    else:
        base_q = 'Transcribe the handwritten text.'
        if use_im_start_end:
            q = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + base_q
        else:
            q = DEFAULT_IMAGE_TOKEN + '\n' + base_q

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], q)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def decode_output(tokenizer, output_ids: torch.Tensor) -> str:
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # Basic cleanup: collapse whitespace
    return ' '.join(text.split())


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    # jiwer metrics operate on sentences; ensure same length lists
    metrics = {}
    metrics['wer'] = wer(refs, preds)
    metrics['cer'] = cer(refs, preds)
    metrics['mer'] = mer(refs, preds)
    metrics['wil'] = wil(refs, preds)
    # Normalized edit distance (character level) - alternative view
    # We'll compute averaged Levenshtein / max(len(ref), 1)
    import numpy as np
    from jiwer import transforms
    # Simpler custom char-level edit distance
    def lev(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        dp = [[0]*(lb+1) for _ in range(la+1)]
        for i in range(la+1): dp[i][0] = i
        for j in range(lb+1): dp[0][j] = j
        for i in range(1, la+1):
            for j in range(1, lb+1):
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
        return dp[la][lb]
    ned = []
    for r, p in zip(refs, preds):
        ned.append(lev(r, p) / max(len(r), 1))
    metrics['normalized_edit_distance'] = float(np.mean(ned))
    return metrics


def save_results(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


@torch.no_grad()
def run_htr_eval(model, tokenizer, image_processor, dataset: List[Dict[str, Any]], image_folder: str, batch_size: int, max_new_tokens: int, output_path: str, num_beams: int = 1, temperature: float = 0.0, chunk_idx: int = 0, num_chunks: int = 1):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
    use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    conv_mode = 'v1'

    # Shard dataset if multi-chunk requested (similar to infer_eval pattern)
    if num_chunks > 1:
        per = (len(dataset) + num_chunks - 1) // num_chunks
        start = chunk_idx * per
        end = min(len(dataset), start + per)
        dataset = dataset[start:end]
        print(f"[INFO] Using shard {chunk_idx}/{num_chunks} with {len(dataset)} samples.")

    # Filter items with required fields and extract ground truth
    valid = []
    for ex in dataset:
        if 'image' not in ex:
            continue
        
        # Extract ground truth from multiple possible locations
        ground_truth = None
        if 'text' in ex:
            ground_truth = ex['text']
        elif 'transcription' in ex:
            ground_truth = ex['transcription']
        elif 'conversations' in ex and len(ex['conversations']) >= 2:
            # Extract from conversations[1]['value'] (GPT response)
            ground_truth = ex['conversations'][1].get('value', '')
        
        if ground_truth:
            ex['text'] = ground_truth
            valid.append(ex)
    
    if not valid:
        print('[WARN] No valid examples with both image and ground truth text found.')
        return

    def batches(lst, bs):
        for i in range(0, len(lst), bs):
            yield lst[i:i+bs]

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rows = []
    preds_all = []
    refs_all = []

    for chunk in tqdm(list(batches(valid, batch_size)), desc='HTR eval', dynamic_ncols=True):
        id_seqs = []
        imgs = []
        refs = []
        for ex in chunk:
            prompt = build_prompt(ex, use_im_start_end, conv_mode)
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').squeeze(0)
            id_seqs.append(ids)
            img_path = ex['image']
            # allow relative path root under image_folder
            if not os.path.isabs(img_path):
                img_path = os.path.join(image_folder, '/'.join(img_path.split('/')[1:])) if '/' in img_path else os.path.join(image_folder, img_path)
            img = Image.open(img_path).convert('RGB')
            px = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
            imgs.append(px)
            refs.append(ex['text'])
        input_ids = pad_sequence(id_seqs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = (input_ids != pad_id).long()
        imgs = torch.stack(imgs).to(device, dtype=next(model.parameters()).dtype)

        generation = model.generate(
            input_ids=input_ids,
            images=imgs,
            attention_mask=attention_mask,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            use_cache=True,
        )

        # We only want the generated continuation, strip the prompt tokens
        for i, out_ids in enumerate(generation):
            # Heuristic: remove prompt portion
            prompt_len = input_ids[i].shape[0]
            gen_ids = out_ids[prompt_len:]
            pred_text = decode_output(tokenizer, gen_ids)
            
            # Save full row with metadata
            result_row = {
                'image': chunk[i]['image'],
                'reference': refs[i],
                'prediction': pred_text
            }
            # Optionally include demographics if present
            if 'Gender' in chunk[i]:
                result_row['gender'] = chunk[i]['Gender']
            if 'WritingType' in chunk[i]:
                result_row['writing_type'] = chunk[i]['WritingType']
            if 'Age' in chunk[i]:
                result_row['age'] = chunk[i]['Age']
            if 'writer_id' in chunk[i]:
                result_row['writer_id'] = chunk[i]['writer_id']
            
            rows.append(result_row)
            preds_all.append(pred_text)
            refs_all.append(refs[i])

    metrics = compute_metrics(preds_all, refs_all)
    print('\n== Handwriting Transcription Metrics ==')
    for k, v in metrics.items():
        print(f'{k}: {v:.4f}')

    if output_path:
        save_results(output_path, rows)
        # also save summary metrics
        with open(output_path + '.metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'[INFO] Saved per-sample results to {output_path} and metrics to {output_path}.metrics.json')


def main():
    ap = argparse.ArgumentParser(description="Handwriting transcription evaluation (LLaVA-Rad style)")
    ap.add_argument('--model_path', required=True, help='Path to fine-tuned LLaVA checkpoint')
    ap.add_argument('--model_base', required=True, help='Path to base Vicuna model (e.g., lmsys/vicuna-7b-v1.5)')
    ap.add_argument('--model_name', default='llavarad')
    ap.add_argument('--data_path', required=True, help='JSON or JSONL dataset')
    ap.add_argument('--loader', default='iam', help='Loader key from llava.utils.data_loaders (or custom)')
    ap.add_argument('--image_folder', required=True)
    ap.add_argument('--prefix', default='')
    ap.add_argument('--run_name', default=None)
    ap.add_argument('--prediction_dir', default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--max_new_tokens', type=int, default=512, help='Maximum tokens to generate (matches radiology report generation)')
    ap.add_argument('--num_beams', type=int, default=1)
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--load_8bit', action='store_true')
    ap.add_argument('--load_4bit', action='store_true')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--chunk_idx', type=int, default=0)
    ap.add_argument('--num_chunks', type=int, default=1)
    ap.add_argument('--output_filename', default='htr_results.jsonl')
    ap.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = ap.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    if args.run_name is None:
        args.run_name = f"{args.prefix}htr_eval"
    if args.prediction_dir is None:
        args.prediction_dir = os.path.join(os.getcwd(), 'results', args.run_name)
    os.makedirs(args.prediction_dir, exist_ok=True)
    output_path = os.path.join(args.prediction_dir, args.output_filename)

    # Load raw dataset first
    raw_items = load_items(args.data_path)

    # Attempt to apply named loader from llava.utils if available
    if args.loader in data_loaders:
        loader_fn = data_loaders[args.loader]
        # Some loaders expect a data_args object; fabricate minimal one if needed
        import inspect
        sig = inspect.signature(loader_fn)
        if 'data_args' in sig.parameters:
            class _DataArgs:
                def __init__(self, data_path):
                    self.data_path = data_path
                    self.split = 'eval'
                    self.fairness_labels = False
            data_args = _DataArgs(args.data_path)
            try:
                loaded = loader_fn(data_args)
            except Exception as e:
                print(f"[WARN] Loader {args.loader} failed ({e}); falling back to raw items.")
                loaded = raw_items
        else:
            try:
                loaded = loader_fn(args.data_path)
            except Exception as e:
                print(f"[WARN] Loader {args.loader} failed ({e}); falling back to raw items.")
                loaded = raw_items
        dataset = loaded
    else:
        print(f"[INFO] Loader {args.loader} not found; using raw items.")
        dataset = raw_items

    # For handwriting datasets, ensure ground truth key normalization to 'text'
    for ex in dataset:
        if 'text' not in ex:
            # Try transcription field
            if 'transcription' in ex:
                ex['text'] = ex['transcription']
            # Try conversations (GPT response in conversations[1]['value'])
            elif 'conversations' in ex and len(ex['conversations']) >= 2:
                ex['text'] = ex['conversations'][1].get('value', '')

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, args.model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit, device=args.device,
    )
    model.to(args.device).eval()

    run_htr_eval(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        dataset=dataset,
        image_folder=args.image_folder,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=output_path,
        num_beams=args.num_beams,
        temperature=args.temperature,
        chunk_idx=args.chunk_idx,
        num_chunks=args.num_chunks,
    )

    print(f"[DONE] Evaluation complete. Results in {output_path}")


if __name__ == '__main__':
    main()
