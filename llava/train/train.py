# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
import random
from typing import Dict, Optional, Sequence, List, Iterable
from peft.utils.other import ModulesToSaveWrapper
import torch
import numpy as np

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, open_image_with_retry
from llava.utils import data_loaders
# from scripts.dac_infereval import load_dac_checkpoint

from PIL import Image, ImageFile
# https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random
from collections import defaultdict
from itertools import cycle

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)

# @torch.no_grad()
# def load_dac_checkpoint(model, dac_path):
#     """Load DAC checkpoint weights into model."""
#     sd = torch.load(dac_path, map_location="cpu")
#     sd = {"base_model.model."+k: v for k, v in sd.items()}
#     sd = {k: v for k, v in sd.items() if k.startswith("base_model.model.dac_modules.")}
#     base = model.state_dict()
#     sd_ok = {k: v for k, v in sd.items() if k in base and v.shape == base[k].shape}    
#     msg = model.load_state_dict(sd_ok, strict=False)
#     print(f"Loaded {len(sd_ok)} DAC parameters from {dac_path}")
#     return len(sd_ok)

@torch.no_grad()
def load_dac_checkpoint(model, dac_path):
    """Load DAC checkpoint weights into model (handles original_module / modules_to_save)."""
    sd = torch.load(dac_path, map_location="cpu")
    # unwrap common wrapper
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # strip DDP prefix if any
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    base = model.state_dict()

    sd_ok = {}

    for k, v in sd.items():
        # expect ckpt keys like: 'dac_modules.<group>.<rest>'
        if not k.startswith("dac_modules."):
            continue
        tail = k.split("dac_modules.", 1)[1]  # '<group>.<rest>'
        if "." not in tail:
            continue
        group, sub = tail.split(".", 1)

        # try both layouts present in your model; also a flat fallback
        for mk in (
            f"base_model.model.dac_modules.{group}.original_module.{sub}",
            f"base_model.model.dac_modules.{group}.modules_to_save.default.{sub}",
            f"base_model.model.dac_modules.{group}.{sub}",
        ):
            if mk in base and base[mk].shape == v.shape:
                sd_ok[mk] = v  # load into every matching location

    msg = model.load_state_dict(sd_ok, strict=False)
    print(f"Loaded {len(sd_ok)} DAC parameters from {dac_path}")
    return len(sd_ok)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_config: Optional[str] = field(default=None)
    vision_tower_checkpoint: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    output_hidden_states: bool = field(default=False)
    fairness_finetune: Optional[bool] = field(default=False,
                                    metadata={"help": "Whether to finetune fairness labels."})
    fairness_config: Optional[str] = field(default=None,
                                        metadata={"help": "Path to the fairness config file."})

    mi_token_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Token subset for MI/CLUB pooling: 'image', 'text', or 'all'. Overrides fairness_config if set."}
    )



@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    loader: str = "default"
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    fairness_labels: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})     


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k and ".modules_to_save." not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_other_trainables_maybe_zero3(named_params):
    def keep(n, p):
        if not p.requires_grad:
            return False
        if "lora_" in n:                      # PEFT handles LoRA
            return False
        if ".modules_to_save." in n:          # PEFT handles CLUB via modules_to_save
            return False

        return True

    return {
        n: maybe_zero_3(p, ignore_status=True, name=n).cpu()
        for n, p in named_params
        if keep(n, p)
    }


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    block_list = ("gender_club", "age_club", "race_club", "handedness_club", "variational_net", "fairness",
                  "mm_projector", "vision_tower", "vision_resampler", "image_processor",
                  "club_modules", "dac_modules")
    for name, module in model.named_modules():
        if any(b in name for b in block_list):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_dac_state_maybe_zero_3(named_params):
    to_return = {k: t for k, t in named_params if "dac_modules" in k}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class BalancedRoundRobinBatchSampler:
    """
    GLOBAL batch sampler: each yielded list has length = per_device_bs * world_size.
    Balances ONE attribute per batch (cycles through attr_order).
    Expects samples with keys in attr_order (e.g., 'anchor_age_group','race_major','gender').
    """
    def __init__(self, list_data_dict: List[dict], global_batch_size: int,
                 attr_order=("anchor_age_group", "race_major", "gender", "handedness"), seed=0):
        self.data = list_data_dict
        self.global_bs = int(global_batch_size)
        self.attr_order = list(attr_order)
        self._attr_cycle = cycle(self.attr_order)
        self.rng = random.Random(seed)
        self.active_attr = None

        self.pools: Dict[str, Dict] = {}
        self.classes: Dict[str, List] = {}

        def norm(v): return v.strip().lower() if isinstance(v, str) else v

        for attr in self.attr_order:
            buckets = defaultdict(list)
            for i, ex in enumerate(self.data):
                if attr in ex:
                    buckets[norm(ex[attr])].append(i)
            pools = {k: v for k, v in buckets.items() if v}
            if not pools:
                raise ValueError(f"No samples for attribute '{attr}'.")
            self.pools[attr] = pools
            self.classes[attr] = sorted(pools.keys(), key=lambda x: str(x))

        def per_class_counts(bs, n):
            base, rem = bs // n, bs % n
            sizes = [base] * n
            for i in range(rem): sizes[i] += 1
            return sizes

        self.takes = {a: per_class_counts(self.global_bs, len(self.classes[a]))
                      for a in self.attr_order}

    def __iter__(self) -> Iterable[List[int]]:
        while True:
            a = next(self._attr_cycle)
            self.active_attr = a
            cls_names = self.classes[a]
            takes = self.takes[a]
            batch_idx = []
            for c, t in zip(cls_names, takes):
                pool = self.pools[a][c]
                batch_idx.extend(self.rng.choices(pool, k=t))  # with replacement
            self.rng.shuffle(batch_idx)
            yield batch_idx

    def __len__(self):
        return max(1, len(self.data) // max(1, self.global_bs))


class ShardBatchSampler:
    """Split each GLOBAL batch into per-rank LOCAL slices."""
    def __init__(self, global_sampler: BalancedRoundRobinBatchSampler,
                 per_device_bs: int, rank: int, world_size: int):
        self.gs = global_sampler
        self.per_device_bs = int(per_device_bs)
        self.rank = int(rank)
        self.world_size = int(max(1, world_size))
        assert self.per_device_bs * self.world_size == self.gs.global_bs

    @property
    def active_attr(self):
        return self.gs.active_attr

    def __iter__(self):
        start = self.rank * self.per_device_bs
        stop = start + self.per_device_bs
        for g in self.gs:
            yield g[start:stop]

    def __len__(self):
        return len(self.gs)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        data_args.data_path = data_path
        list_data_dict = data_loaders[data_args.loader](data_args)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = open_image_with_retry(os.path.join(image_folder, image_file))
            if image is None:
                logging.error("Use an empty image.")
                image = Image.new('RGB', (224, 224), tuple(int(x*255) for x in processor.image_mean))
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        # If needs to return fairness labels
        if self.data_args.fairness_labels:
            if 'gender' in self.list_data_dict[i]:
                data_dict['gender'] = self.list_data_dict[i]['gender']
            if "Gender" in self.list_data_dict[i]:
                data_dict['gender'] = self.list_data_dict[i]['Gender']
            if 'anchor_age_group' in self.list_data_dict[i]:
                data_dict['anchor_age_group'] = self.list_data_dict[i]['anchor_age_group']
            if 'race_major' in self.list_data_dict[i]:
                data_dict['race_major'] = self.list_data_dict[i]['race_major']
            if 'WritingType' in self.list_data_dict[i]:
                data_dict['handedness'] = self.list_data_dict[i]['WritingType']
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if "gender" in instances[0]:
            gmap = {"male": 0, "m": 0, "man": 0, "female": 1, "f": 1, "woman": 1, "w": 1} # update as needed for every dataset
            genders = [inst.get("gender") for inst in instances]
            # print(f"\n[DEBUG] Raw gender values: {genders[:4]}")  # First 10
            # print(f"[DEBUG] Unique raw values: {set(genders)}")
            g_ids = [gmap.get(str(g or "").strip().lower(), 3) for g in genders] # should never get 3, please clean otherwise will get an erro in the classifier
            # print(f"[DEBUG] Encoded gender IDs: {g_ids[:10]}")
            # print(f"[DEBUG] Class distribution: {torch.bincount(torch.tensor(g_ids), minlength=2)}")
            batch["gender"] = torch.tensor(g_ids, dtype=torch.long)

        if "race_major" in instances[0]:
            rmap = {
                "white": 0, "black": 1, "black or african american": 1, "asian": 2, "hispanic": 3, 
                "hispanic or latino": 3, "other": 4, "american indian or alaska native": 5, 
                "native hawaiian or pacific islander": 6, "unknown": 7, "": 7, "declined / unable to obtain":  7
            }
            races = [inst.get("race_major") for inst in instances]
            r_ids = [rmap.get(str(r or "").strip().lower(), 7) for r in races]
            batch["race_major"] = torch.tensor(r_ids, dtype=torch.long)

        if "anchor_age_group" in instances[0]:
            anchor_age_groups = [inst.get("anchor_age_group") for inst in instances]
            batch["anchor_age_group"] = torch.tensor(anchor_age_groups, dtype=torch.long)
        
        if "handedness" in instances[0]:
            hmap = {"right": 0, "right-handed": 0, "left": 1, "left-handed": 1}
            handednesses = [inst.get("handedness") for inst in instances]
            h_ids = [hmap.get(str(h or "").strip().lower(), 3) for h in handednesses]
            batch["handedness"] = torch.tensor(h_ids, dtype=torch.long)

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)



def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    # Set seed for reproducibility
    set_seed(training_args.seed)
    rank0_print(f"Setting random seed to {training_args.seed}")
    
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            if model_args.output_hidden_states is not None:
                config.output_hidden_states = model_args.output_hidden_states
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                output_hidden_states=model_args.output_hidden_states,
                cache_dir=training_args.cache_dir,
                fairness_finetune=model_args.fairness_finetune,
                fairness_config=model_args.fairness_config,
                mi_token_mode=model_args.mi_token_mode,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            output_hidden_states=model_args.output_hidden_states,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
        AssertionError("4/8-bit training needs to add Modules to save")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        modules_to_save = []
        if model_args.fairness_finetune:
            if hasattr(model, "club_modules") and isinstance(model.club_modules, torch.nn.ModuleDict):
                club_keys = list(model.club_modules.keys())
                modules_to_save += [f"club_modules.{k}" for k in club_keys]
                rank0_print(f"modules_to_save (CLUB): {modules_to_save}")
            else:
                rank0_print("WARN: model.club_modules not found or not a ModuleDict; skipping modules_to_save for CLUB.")
            
            # Add DAC modules to modules_to_save if we want to finetune them during debias
            if hasattr(model, "dac_modules") and isinstance(model.dac_modules, torch.nn.ModuleDict):
                dac_frozen = getattr(model, "dac_frozen", True)
                fairness_stage = getattr(model, "fairness_stage", "")
                # Only add DAC to modules_to_save if we're in debias stage AND dac_frozen=False
                if fairness_stage in ["debias", "debias_joint"] and not dac_frozen:
                    dac_keys = list(model.dac_modules.keys())
                    modules_to_save += [f"dac_modules.{k}" for k in dac_keys]
                    rank0_print(f"modules_to_save (DAC): {[f'dac_modules.{k}' for k in dac_keys]}")
                    rank0_print(f"[DAC] Will finetune DAC during {fairness_stage} (dac_frozen=False)")
                elif fairness_stage in ["debias", "debias_joint"]:
                    rank0_print(f"[DAC] Will freeze DAC during {fairness_stage} (dac_frozen=True)")

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            modules_to_save=modules_to_save,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False) # We do not need to train the fairness club
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if getattr(model, "fairness_finetune", False) \
    and getattr(model, "debiasing_method", "") in ["dac_qkv", "dac_mlp"] \
    and getattr(model, "fairness_stage", "") in ["debias", "debias_joint"]:
        dac_path = os.path.join(model_args.model_name_or_path, "dac_modules.bin")
        dac_frozen = getattr(model, "dac_frozen", True)
        
        if os.path.isfile(dac_path):
            loaded = load_dac_checkpoint(model, dac_path)  # safer loader
            rank0_print(f"[DAC] Loaded {loaded} tensors from {dac_path}")
            
            # Only freeze if dac_frozen=True
            if dac_frozen:
                for head in model.dac_modules.values():
                    head.eval()
                    for p in head.parameters():
                        p.requires_grad_(False)
                rank0_print("[DAC] Frozen DAC modules for adversarial training (dac_frozen=True)")
            else:
                for head in model.dac_modules.values():
                    head.train()
                    for p in head.parameters():
                        p.requires_grad_(True)
                rank0_print("[DAC] DAC modules are trainable for joint adversarial training (dac_frozen=False)")
        else:
            rank0_print(f"[DAC] WARNING: No dac_modules.bin found at {dac_path}; debiasing needs pretrained DAC.")      
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    # print("==============================================================\n==============================================")
    # print(getattr(model, "is_loaded_in_4bit", False), getattr(model, "is_loaded_in_8bit", False))
    
    # # 1) Make sure 4-bit Linear modules are present
    # import bitsandbytes as bnb
    # has_4bit = any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    # print("has Linear4bit modules:", has_4bit)

    # # 2) Peek one Linear4bit to see its compute dtype (should be bf16/fp16)
    # lin4 = next(m for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
    # print("Linear4bit compute dtype:", lin4.compute_dtype)

    # # 3) LoRA & projector should be in (b)float and trainable
    # trainable = [(n, p.dtype) for n, p in model.named_parameters() if p.requires_grad][:5]
    # print("sample trainable params (name, dtype):", trainable)

    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args, 
                    **data_module)

    # Verify CLUB params are not frozen by PEFT.
    if getattr(model, "debiasing_method", None) == "mi_club" and getattr(model, "fairness_stage", None) == "debias":
        club_container = getattr(model, "club_modules", None)
        if club_container is None and hasattr(model, "get_model"):
            base = model.get_model()
            club_container = getattr(base, "club_modules", None)

        active = getattr(model, "active_adapter", "default")
        assert club_container is not None, "club_modules not found."

        bad = []
        for key, mod in club_container.items():
            if not isinstance(mod, ModulesToSaveWrapper):
                raise RuntimeError(f"club_modules.{key} is not wrapped by ModulesToSaveWrapper (modules_to_save).")
            try:
                train_mod = mod.modules_to_save[active]
            except KeyError:
                raise RuntimeError(f"club_modules.{key} has no modules_to_save entry for adapter '{active}'.")
            frozen = [n for n, p in train_mod.named_parameters() if not p.requires_grad]
            if frozen:
                bad += [f"club_modules.{key}.{n}" for n in frozen]
        assert not bad, f"Frozen CLUB trainables (should be True): {bad}"
    elif getattr(model, "fairness_finetune", False) \
                and getattr(model, "debiasing_method", "") in ["dac_qkv", "dac_mlp"] \
                and getattr(model, "fairness_stage", "") == "pretrain_dac":
        for p in model.parameters():
            p.requires_grad_(False)

        for head in model.dac_modules.values():
            for p in head.parameters():
                p.requires_grad_(True)

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert all(n.startswith("dac_modules") for n in trainable), f"Unexpected trainables: {trainable}"
        training_args.lora_enable = False

    # n_trainable = 0
    # for n,p in model.named_parameters():
    #     if n.startswith("dac_modules."):
    #         print(n, "requires_grad=", p.requires_grad)
    #         n_trainable += int(p.requires_grad)
    # print("[CHECK] #trainable DAC tensors:", n_trainable)

    # trainer.create_optimizer()  # forces creation now
    # names_in_opt = set()
    # for g in trainer.optimizer.param_groups:
    #     for p in g["params"]:
    #         names_in_opt.update([n for n,pp in trainer.model.named_parameters() if pp is p])
    # print("[CHECK] DAC in optimizer:", any(n.startswith("dac_modules.") for n in names_in_opt))
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        # missing, unexpected = model.load_state_dict(torch.load("/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/checkpoints/biomedclip_cxr_518-lora-3e-1e-4-20251118022545/club_age_variational_net.pt"), strict=False) # TODO remove, just for ablation - Mahesh
        # print(missing, unexpected)
        trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        if getattr(model, "fairness_finetune", False):
            # Persist debias/fairness knobs into config.json for reproducibility.
            # Prefer the canonical names used in fairness configs + model code, but keep legacy aliases
            # where they existed in older runs.
            f = {
                "debiasing_method": getattr(model, "debiasing_method", None),
                "gender_num": getattr(model, "gender_num", None),
                "age_num": getattr(model, "age_num", None),
                "race_num": getattr(model, "race_num", None),
                "handedness_num": getattr(model, "handedness_num", None),
                "target_dem": getattr(model, "target_dem", []),
                "fairness_finetune": getattr(model, "fairness_finetune", None),

                "gender_lmda": getattr(model, "gender_lmda", None),
                "age_lmda": getattr(model, "age_lmda", None),
                "race_lmda": getattr(model, "race_lmda", None),
                "handedness_lmda": getattr(model, "handedness_lmda", None),

                # Canonical names
                "mi_club_training_weight": getattr(model, "mi_club_training_weight", None),
                "ce_loss_weight": getattr(model, "ce_loss_weight", None),
                "mi_drop_bos_eos": getattr(model, "mi_drop_bos_eos", None),
                "hidden_depth": getattr(model, "hidden_depth", None),
                "mi_token_mode": getattr(model, "mi_token_mode", None),
                "tau": getattr(model, "tau", None),

                "fairness_stage": getattr(model, "fairness_stage", None),
                "tapped_layers": getattr(model, "tapped_layers", None),
                "fairness_lambda": getattr(model, "fairness_lambda", None),

                "dac_frozen": getattr(model, "dac_frozen", None),
                "dac_loss_mode": getattr(model, "dac_loss_mode", None),
                "dac_nhead": getattr(model, "dac_nhead", None),
                "dac_dropout": getattr(model, "dac_dropout", None),
                "dac_mlp_ratio": getattr(model, "dac_mlp_ratio", None),
                "dac_query_mode": getattr(model, "dac_query_mode", None),
                "dac_pooling_mode": getattr(model, "dac_pooling_mode", None),

                "dem_class_counts": getattr(model, "dem_class_counts", None),
                "focal_loss_gamma": getattr(model, "focal_loss_gamma", None),
                "focal_loss_alpha": getattr(model, "focal_loss_alpha", None),

                # Legacy aliases (kept for older tooling / downstream scripts)
                "mi_club_train_weight": getattr(model, "mi_club_training_weight", None),
                "mi_ce_loss_weight": getattr(model, "ce_loss_weight", None),
                "dac_nheads": getattr(model, "dac_nheads", None) if hasattr(model, "dac_nheads") else getattr(model, "dac_nhead", None),
                "dac_class_counts": getattr(model, "dac_class_counts", None),
                "dac_query_mode": getattr(model, "dac_query_mode", None),
                "dac_mlp_ratio": getattr(model, "dac_mlp_ratio", None),
                "dac_dropout": getattr(model, "dac_dropout", None),

                # Explicitly persist if present on the model
                "kd_alpha": getattr(model, "kd_alpha", None),
                "kd_temperature": getattr(model, "kd_temperature", None),
            }
            model.config.fairness = {k: v for k, v in f.items() if v is not None}
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir)
            if non_lora_state_dict:
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
    if trainer.is_world_process_zero() and getattr(model, "debiasing_method", None) in ["dac_qkv", "dac_mlp"]:
        dac_state = get_dac_state_maybe_zero_3(model.named_parameters())
        if dac_state:
            torch.save(dac_state, os.path.join(training_args.output_dir, "dac_modules.bin"))


if __name__ == "__main__":
    train()
