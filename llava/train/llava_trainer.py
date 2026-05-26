from logging import warning
import os
import torch

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional
from transformers.trainer import _is_peft_model 
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from contextlib import contextmanager
from torch.nn import functional as F
from llava.constants import IGNORE_INDEX
# from transformers import TrainerCallback

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k and ".modules_to_save." not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

    
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


# class SubLossLogger(TrainerCallback):
#     def __init__(self): self.cache = None
#     def on_step_end(self, args, state, control, model=None, **kwargs):
#         # pick up latest sub-losses from the model (attach them on the model in compute_loss)
#         self.cache = getattr(model, "_latest_sublosses", None)
#     def on_log(self, args, state, control, logs=None, model=None, **kwargs):
#         if self.cache:
#             out = {f"train/{k}": (v.detach().float().mean().item() if torch.is_tensor(v) else float(v))
#                    for k, v in self.cache.items()}
#             # merge into existing logs dict
#             logs.update(out)

class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
    

def _resolve_base_llava(model_like):
    """
    Return the actual LlavaLlamaForCausalLM instance no matter whether
    we're wrapped by DeepSpeed or PEFT.
    """
    m = getattr(model_like, "module", model_like)  # deepspeed.Engine -> .module
    # PEFT models expose get_base_model()
    get_base = getattr(m, "get_base_model", None)
    if callable(get_base):
        return get_base()
    base = getattr(m, "base_model", m)
    base = getattr(base, "model", base)
    return base  # should be LlavaLlamaForCausalLM


def _resolve_peft_wrapper(model_like):
    """
    Return the top-level object that owns PEFT's adapter methods.
    This is typically the PEFT wrapper (PeftModel); under DeepSpeed that's engine.module.
    """
    return getattr(model_like, "module", model_like)


@contextmanager
def fairness_disabled(model_like):
    """
    Temporarily set base.fairness_finetune = False so your model's forward()
    skips the debias branch. Works with DeepSpeed + PEFT.
    """
    base = _resolve_base_llava(model_like)
    prev = getattr(base, "fairness_finetune", False)
    base.fairness_finetune = False
    try:
        yield
    finally:
        base.fairness_finetune = prev


@contextmanager
def lora_disabled(model_like):
    """
    Temporarily disable ALL LoRA adapters during the block.

    Preferred path: use PEFT's built-in `disable_adapter()` context manager.
    Fallback path: zero/restore internal scaling safely if PEFT API isn't available.
    """
    wrapper = _resolve_peft_wrapper(model_like)
    disable_cm = getattr(wrapper, "disable_adapter", None)

    if callable(disable_cm):
        # PEFT-native and safest
        with disable_cm():
            yield
        return

    # Should be used with caution: directly manipulate internal scaling factors.
    saved = []
    warning("lora_disabled: PEFT disable_adapter() not found; using manual scaling zeroing.")
    try:
        for mod in wrapper.modules():
            if hasattr(mod, "scaling"):
                saved.append((mod, mod.scaling))
                s = mod.scaling
                if isinstance(s, dict):
                    mod.scaling = {k: 0.0 for k in s.keys()}
                else:
                    mod.scaling = 0.0
        yield
    finally:
        for mod, s in saved:
            mod.scaling = s


@contextmanager
def teacher_mode(model_like):
    """
    A clean "teacher pass" context:
      - eval() (no dropout)
      - no grads (no graph, regardless of requires_grad flags)
      - LoRA disabled
      - fairness branch disabled
    Works whether you pass the DeepSpeed engine or the PEFT/HF model.
    """
    wrapper = _resolve_peft_wrapper(model_like)
    was_training = wrapper.training
    wrapper.eval()

    try:
        with torch.no_grad():
            with fairness_disabled(model_like):
                with lora_disabled(model_like):
                    yield
    finally:
        if was_training:
            wrapper.train()


class LLaVATrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_log_step = -1
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Matches HF default behavior, with two additions:
          - if label_smoother is used, add back MI penalty from outputs["losses"]["loss_mi_total"]
          - log sub-losses in outputs["losses"] (if present)
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # p = self.state.global_step / max(1, getattr(self.state, "max_steps", 1 + self.state.global_step))
        # start, end = 0.01, 0.90 # 1% to 90% of training
        # lam_max = float(getattr(getattr(model, "module", model), "fairness_lambda", 0.8)) 
        # lam_min = 0.1
        # if p <= start:
        #     lam = lam_min
        # elif p >= end:
        #     lam = lam_max
        # else:
        #     lam = lam_max # TODO remove this later for the scheduler
        #     lam = max(lam_max * (p - start) / (end - start), lam_min)
        # outputs = model(**inputs, cur_lam=lam)
        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Also add custom losses for the logging
        sub = (outputs.get("losses") if isinstance(outputs, dict)
               else getattr(outputs, "losses", None))
        if sub is not None and "loss_mi_total" in sub and self.label_smoother is not None:
                mi = sub["loss_mi_total"]
                if torch.is_tensor(mi):
                    loss = loss + mi
                else:
                    loss = loss + float(mi)
        
        if sub is not None and "loss_dac_total" in sub and self.label_smoother is not None:
                dac_loss = sub["loss_dac_total"]
                if torch.is_tensor(dac_loss):
                    loss = loss + dac_loss
                else:
                    loss = loss + float(dac_loss)

        # Compute KD loss
        # mroot = getattr(model, "module", model)
        # p = self.state.global_step / max(1, getattr(self.state, "max_steps", self.state.global_step + 1))
        # start_ratio = float(getattr(mroot, "kd_start_ratio", 0.45))  # start KD at 40% of training
        # end_ratio   = float(getattr(mroot, "kd_end_ratio",   0.6))  # end KD at 60%
        # ramp_ratio  = float(getattr(mroot, "kd_ramp_ratio",  0.05))  # 5% linear ramps


        # if p < start_ratio:
        #     w = 0.0
        # elif p < start_ratio + ramp_ratio:
        #     w = (p - start_ratio) / max(ramp_ratio, 1e-8)  # ramp up 0→1
        # elif p < end_ratio - ramp_ratio:
        #     w = 1.0
        # elif p < end_ratio:
        #     w = (end_ratio - p) / max(ramp_ratio, 1e-8)    # ramp down 1→0
        # else:
        #     w = 0.0
        # # w = 0.0
        # if w != 0.0:
        #     with teacher_mode(model):
        #         teacher_out = model(
        #             input_ids=inputs["input_ids"],
        #             attention_mask=inputs.get("attention_mask"),
        #             images=inputs.get("images"),
        #             labels=None,                
        #             no_ce_loss=True,            
        #             output_hidden_states=False,  
        #             return_dict=True,
        #         )

        #     # We should use labels from the student model's inputs
        #     T = float(getattr(mroot, "kd_temperature", 2.0))
        #     alpha = float(getattr(mroot, "kd_alpha", 0.2))
        #     s_logits = outputs.logits[:, :-1, :]          # [B, Ts-1, V]
        #     t_logits = teacher_out.logits[:, :-1, :]      # [B, Tt-1, V]
        #     lab_shift = outputs["labels"][:, 1:]           # [B, L_lab]

        #     # crop everything to CE’s supervised window length
        #     L = min(s_logits.size(1), t_logits.size(1), lab_shift.size(1))
        #     s_logits = s_logits[:, :L, :]
        #     t_logits = t_logits[:, :L, :]
        #     mask = (lab_shift[:, :L] != IGNORE_INDEX).float()   # same mask CE uses via ignore_index

        #     # KD
        #     s = (s_logits / T).float()
        #     t = (t_logits / T).float()
        #     logp_s = F.log_softmax(s, dim=-1)
        #     logp_t = F.log_softmax(t, dim=-1)
        #     kl_tok = F.kl_div(logp_s, logp_t, log_target=True, reduction="none").sum(dim=-1)
            
        #     kd_loss = (kl_tok * mask).sum() / mask.sum().clamp_min(1.0)
        #     # kd_loss = kd_loss * (alpha * T * T)
        
        #     # add scheduler weights and add kdloss in the main loss
        #     alpha_eff = alpha * w
        #     kd_loss = kd_loss * (alpha_eff * T * T)
        #     loss = loss + kd_loss
            
        #     if hasattr(outputs, "losses"):
        #         outputs.losses["loss_kd"] = kd_loss.detach()

        if (sub
            and self.is_world_process_zero()
            and self.state.global_step > 0
            and self.state.global_step % self.args.logging_steps == 0
            and self.state.global_step != self._last_log_step):
            to_log = {k: (v.detach().float().mean().item() if torch.is_tensor(v) else float(v))
                      for k, v in sub.items()}
            self.log({f"train/{k}": v for k, v in to_log.items()})
            self._last_log_step = self.state.global_step

        return (loss, outputs) if return_outputs else loss
    
    # def get_train_dataloader(self):
    #     # Enable when data_args.balance_mode == "roundrobin" (or balanced_roundrobin=True)
    #     da = getattr(self.train_dataset, "data_args", None)
    #     use_balanced = False
    #     if da is not None:
    #         mode = str(getattr(da, "balance_mode", "")).lower()
    #         use_balanced = (mode == "roundrobin") or bool(getattr(da, "balanced_roundrobin", False))
    #     if not use_balanced:
    #         return super().get_train_dataloader()

    #     per_device_bs = self.args.per_device_train_batch_size
    #     world_size    = max(1, self.args.world_size)
    #     rank          = self.args.process_index
    #     seed          = int(getattr(da, "seed", getattr(self.args, "seed", 0)))
    #     attr_order    = tuple(getattr(da, "balance_attrs", ("anchor_age_group","race_major","gender")))
    #     global_bs     = per_device_bs * world_size

    #     # works with LazySupervisedDataset which has list_data_dict
    #     list_data = getattr(self.train_dataset, "list_data_dict", None) or self.train_dataset

    #     base  = BalancedRoundRobinBatchSampler(list_data, global_bs, attr_order=attr_order, seed=seed)
    #     shard = ShardBatchSampler(base, per_device_bs=per_device_bs, rank=rank, world_size=world_size)

    #     # let compute_loss know which attribute is active (optional)
    #     self._active_attr_source = shard

    #     return DataLoader(
    #         self.train_dataset,
    #         batch_sampler=shard,                       # <-- batch_sampler, not sampler/batch_size
    #         collate_fn=self.data_collator,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #         persistent_workers=self.args.dataloader_persistent_workers,
    #     )
        
    def _save_checkpoint(self, model, trial, metrics=None): 
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        if getattr(self.model, "fairness_stage") == "pretrain_dac" and getattr(self.model, "fairness_finetune", False):
            engine = getattr(self, "deepspeed", None)
            if engine is None and hasattr(self, "model_wrapped"):
                engine = self.model_wrapped  # HF wraps the model with DS engine
            if engine is not None and hasattr(engine, "save_checkpoint"):
                engine.save_checkpoint(output_dir, tag=f"global_step{self.state.global_step}")
            self._save_rng_state(output_dir)
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            self.state.save_to_json(os.path.join(run_dir, "trainer_state.json"))

            other_sd = get_peft_state_non_lora_maybe_zero_3(model.named_parameters(), require_grad_only=True)
            if other_sd and (self.args.local_rank in (0, -1)):
                torch.save(other_sd, os.path.join(output_dir, "dac_modules.bin"))
                unwrap_model(self.model).config.save_pretrained(output_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(output_dir)
            return
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            other_sd = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
            if other_sd and (self.args.local_rank in (0, -1)):
                torch.save(other_sd, os.path.join(output_dir, 'non_lora_trainables.bin'))
            unwrap_model(self.model).config.save_pretrained(output_dir)
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
