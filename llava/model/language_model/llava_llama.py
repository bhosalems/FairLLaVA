#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..fairness import DACMLP, CLUBForCategorical, DACQKV
from torch.func import functional_call
import torch.nn.functional as F
from torch.autograd import Function

# Gradient Reversal Layer for adversarial debiasing
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, lambda_param: float):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_param * grad_output, None

def class_balanced_weights(counts, beta=0.999994, device=None, dtype=torch.float):
    """
    Compute class-balanced weights using effective number of samples.
    
    Args:
        counts: list/array of sample counts per class
        beta: decay factor (default: 0.9999)
        device: target device
        dtype: target dtype (calculations done in float32 for stability)
    
    Returns:
        w: normalized class weights
        priors: class prior probabilities
    """
    c = torch.tensor(counts, dtype=torch.float32, device=device)
    beta_c = torch.pow(beta, c).clamp(min=1e-8, max=1.0 - 1e-8)
    w = (1. - beta) / (1. - beta_c).clamp(min=1e-8)
    w = w / w.sum() * len(c)
    priors = (c / c.sum()).clamp(min=1e-8)
    return w.to(dtype), priors.to(dtype)

def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        logits: [B, num_classes] raw logits from the model
        targets: [B] ground truth class indices
        alpha: [num_classes] or float - weighting factor for each class (optional)
        gamma: float - focusing parameter (default: 2.0)
        reduction: 'mean' or 'sum' or 'none'
    
    Returns:
        loss: scalar or [B] depending on reduction
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # probability of true class
    focal_weight = (1 - pt) ** gamma
    
    if alpha is not None:
        if isinstance(alpha, (list, tuple)):
            alpha = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
        if alpha.dim() == 1:
            alpha_t = alpha[targets]
        else:
            alpha_t = alpha
        focal_loss = alpha_t * focal_weight * ce_loss
    else:
        focal_loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss
class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class TokenPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.scorer = nn.Linear(d, 1, bias=False)
    def forward(self, H, mask):  # H:[B,T,D], mask:[B,T]
        scores = self.scorer(H).squeeze(-1)                 # [B,T]
        scores = scores.masked_fill(mask==0, float('-inf'))
        attn = scores.softmax(dim=1)                        # [B,T]
        return (H * attn.unsqueeze(-1)).sum(1)              # [B,D]


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(
        self,
        config,
        fairness_finetune: bool = False,
        fairness_config: Optional[str] = None,
        mi_token_mode: Optional[str] = None,
    ):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.fairness_finetune = bool(
            fairness_finetune or getattr(config, "fairness_finetune", False)
        )
        self.fairness_stage = ""
        self.ce_loss_weight = 1.0
        if self.fairness_finetune:
            # Load fairness config if provided
            fcfg = None
            if fairness_config is not None:
                import json
                with open(fairness_config, 'r') as f:
                    fcfg = json.load(f)
            elif hasattr(config, "fairness") and isinstance(config.fairness, dict):
                # Load from config if present
                fcfg = dict(config.fairness)
            
            fcfg = fcfg or {}
            # Generic parameters
            self.gender_num = fcfg.get("gender_num", 2)
            self.age_num = fcfg.get("age_num", 3)
            self.race_num = fcfg.get("race_num", 9)
            self.handedness_num = fcfg.get("handedness_num", 2)
            self.target_dem = fcfg.get("target_dem", ["race", "gender", "race", "handedness"])

            # DAC parameters
            self.num_hidden_layers = fcfg.get("num_hidden_layers", 32)
            self.tapped_layers = fcfg.get("tapped_layers", [16])
            self.dac_loss_mode = fcfg.get("dac_loss_mode", "ce") # "ce" | "focal" | "adversarial" | "kl_uniform"
            self.dac_nhead = fcfg.get("dac_nhead", 4)
            self.dac_dropout = fcfg.get("dac_dropout", 0.1)
            self.dac_mlp_ratio = fcfg.get("dac_mlp_ratio", 0.5)
            self.fairness_stage = fcfg.get("fairness_stage", "debias")  # "pretrain_dac" | "debias" | "debias_joint"
            self.dac_query_mode = fcfg.get("dac_query_mode", "learned") # "learned" | "conditioned" | "multi"
            self.dac_frozen = fcfg.get("dac_frozen", False)  # Whether to freeze DAC during debiasing
            
            # Adversarial training parameters
            self.adversarial_dac_weight = fcfg.get("adversarial_dac_weight", 1.0)
            self.adversarial_encoder_weight = fcfg.get("adversarial_encoder_weight", 0.1)
            # Global scaling for the fairness loss term when combining with CE.
            # Default to 1.0 to preserve legacy behavior when this was effectively unscaled.
            self.fairness_lambda = fcfg.get("fairness_lambda", 1.0)

            # Focal loss parameters (for handling class imbalance)
            self.focal_loss_gamma = fcfg.get("focal_loss_gamma", 2.0)
            self.focal_loss_alpha = fcfg.get("focal_loss_alpha", None)  # Can be None, float, or list per class
            
            # MI loss parammeters
            self.race_lmda = fcfg.get("race_lmda", 0.2)
            self.age_lmda = fcfg.get("age_lmda", 0.6)
            self.gender_lmda = fcfg.get("gender_lmda", 0.1)
            self.handedness_lmda = fcfg.get("handedness_lmda", 0.1)
            self.mi_club_training_weight = fcfg.get("mi_club_training_weight", 1.0)
            self.ce_loss_weight = fcfg.get("ce_loss_weight", 1.0)
            self.hidden_depth = fcfg.get("hidden_depth", 0.5) 
            self.dem_class_counts = fcfg.get("dem_class_counts", None) # Optional class weights for CE loss
            self.tau = fcfg.get("tau", 0.5)  # Temperature for logit adjustment

            # Which tokens to use when computing MI/CLUB terms.
            # - "image": image feature tokens only (current default behavior)
            # - "text":  text tokens only (valid tokens excluding image features)
            # - "all":   all valid tokens (image + text)
            self.mi_token_mode = (mi_token_mode or fcfg.get("mi_token_mode", "image")).lower()

            # Optional: drop BOS/EOS positions from the MI mask.
            # This is a heuristic (position-based), but works well for typical LLM tokenization.
            # Default True to avoid putting MI pressure on boundary/special tokens.
            self.mi_drop_bos_eos = bool(fcfg.get("mi_drop_bos_eos", True))

            # Knowledge distillation parameters
            self.kd_alpha = fcfg.get("kd_alpha", 0.2)
            self.kd_temperature = fcfg.get("kd_temperature", 2.0)

            # Build classifier or modules for debiasing
            self.debiasing_method = fcfg.get("debiasing_method", "dac_qkv")
            
            if self.tapped_layers == -1:
                self.tapped_layers = list(range(self.num_hidden_layers))
            
            if self.debiasing_method == "dac_qkv":
                self.dac_modules = nn.ModuleDict()
                for dem in self.target_dem:
                    self.dac_modules[dem] = DACQKV(
                        d_model=config.hidden_size,
                        label_num=getattr(self, f"{dem}_num"),
                        n_layers_tapped=len(self.tapped_layers),
                        n_heads=self.dac_nhead,
                        mlp_ratio=self.dac_mlp_ratio,
                        dropout=self.dac_dropout,
                        query_mode=self.dac_query_mode,
                    )
                if self.fairness_stage in ["debias", "debias_joint"] and self.dac_frozen:
                    for head in self.dac_modules.values():
                        head.eval()
                        for p in head.parameters(): 
                            p.requires_grad_(False)
            
            elif self.debiasing_method == "dac_mlp":
                self.dac_modules = nn.ModuleDict()
                for dem in self.target_dem:
                    self.dac_modules[dem] = DACMLP(
                        d_model=config.hidden_size,
                        label_num=getattr(self, f"{dem}_num"),
                        n_layers_tapped=len(self.tapped_layers),
                        mlp_ratio=self.dac_mlp_ratio,
                        dropout=self.dac_dropout,
                        use_layer_pos=True,
                        pooling_mode=fcfg.get("dac_pooling_mode", "mean"),  # "mean" | "max" | "weighted"
                    )
                if self.fairness_stage in ["debias", "debias_joint"] and self.dac_frozen:
                    for head in self.dac_modules.values():
                        head.eval()
                        for p in head.parameters(): 
                            p.requires_grad_(False)
            
            elif self.debiasing_method == "mi_club":
                self.club_modules = nn.ModuleDict()
                for dem in self.target_dem:
                    self.club_modules[dem] = CLUBForCategorical(input_dim=config.hidden_size, label_num=getattr(self, f"{dem}_num"))
            else:
                raise ValueError(f"Unknown debiasing method {self.debiasing_method}")
        # Initialize weights and apply final processing
        self.post_init()

    def _pool_hidden(self, hidden_states, attention_mask):
       # Average pooling 
       # hidden_states: (B, T, H), attention_mask: (B, T)
        if attention_mask is None:
            attention_mask = hidden_states.new_ones(hidden_states.size()[:-1])
        denom = attention_mask.sum(dim=1, keepdim=True, dtype=hidden_states.dtype).clamp_min(1.0)
        z = (hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1, dtype=hidden_states.dtype) / denom
        return z  # (B, H)
    
    def _pool_hidden2(self, hidden_states, mask): # this worked better 
        # Random dynamic pooling over tokens (images and text)
        # hidden_states: (B, T, H); attention_mask: (B, T) with 1 for valid tokens
        if mask is None:
            mask = hidden_states.new_ones(hidden_states.size()[:-1])
        probs = mask.float()
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)  # per-sample categorical
        idx = torch.multinomial(probs, num_samples=1).squeeze(1)   # (B,)
        z = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), idx]
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        return z  # (B, H)

    def _select_mi_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        image_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Builds the token mask used for MI/CLUB pooling.

        Returns a mask shaped [B, T] with non-zero entries indicating tokens to consider.
        """
        if attention_mask is None and image_mask is None:
            return None

        am = None
        if attention_mask is not None:
            am = attention_mask.bool() if attention_mask.dtype != torch.bool else attention_mask

        im = None
        if image_mask is not None:
            im = image_mask.bool() if image_mask.dtype != torch.bool else image_mask

        mode = getattr(self, "mi_token_mode", "image")
        if mode == "image":
            mi_mask = im
        elif mode == "text":
            mi_mask = (am & (~im)) if (am is not None and im is not None) else am
        elif mode == "all":
            mi_mask = am
        else:
            raise ValueError(f"Unknown mi_token_mode={mode!r}. Use one of: 'image', 'text', 'all'.")

        # Optionally drop BOS/EOS positions (first token, last valid token), NA for image masks.
        if mode != "image" and mi_mask is not None and mi_mask.dim() == 2 and getattr(self, "mi_drop_bos_eos", False):
            # Drop BOS at position 0 (if present)
            mi_mask = mi_mask.clone()
            mi_mask[:, 0] = False

            # Drop EOS at last valid position per sample.
            # Prefer attention_mask to identify valid positions; fall back to current mi_mask.
            base = am if am is not None else mi_mask
            lengths = base.long().sum(dim=1)  # number of valid tokens
            last_idx = (lengths - 1).clamp_min(0)
            mi_mask[torch.arange(mi_mask.size(0), device=mi_mask.device), last_idx] = False

        # Safety: if mask ends up empty for any sample, fall back to attention_mask (or all-ones if missing)
        if mi_mask is not None:
            if mi_mask.dim() != 2:
                return mi_mask
            row_sum = mi_mask.sum(dim=1)
            if (row_sum == 0).any():
                if am is not None:
                    mi_mask = am
                else:
                    mi_mask = torch.ones_like(mi_mask, dtype=torch.bool)
        return mi_mask
    
    def _masked_mean(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H:    [B, T, D], hidden states for one layer
        mask: [B, T] with 1 for tokens to include (e.g., image tokens), else 0
        returns [B, D]
        """
        if mask is None:
            return H.mean(dim=1)
        w = mask.to(dtype=H.dtype)    
        denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (H * w.unsqueeze(-1)).sum(dim=1) / denom

    def _collect_layer_summaries(self, hidden_states, image_mask, tap_layers):
        """
        hidden_states: tuple/list of [B,T,D] (ensure output_hidden_states=True upstream)
        tap_layers: list[int] indices into hidden_states to summarize
        returns X: [B, L_taps, D]
        """
        feats = []
        for li in tap_layers:
            H = hidden_states[li]                   # [B,T,D]
            e = self._masked_mean(H, image_mask)    # [B,D] # # TODO Try token pooler class because it is more flexible.
            feats.append(e)
        return torch.stack(feats, dim=1)            # [B, L_taps, D]
    
    def _mi_term(self, z, club_module, lmda, lbl):
        if club_module is None or lmda <= 0.0 or lbl is None:
            return z.new_zeros(())
        assert z.requires_grad, "MI term needs z that requires grad."
        # Treat CLUB params as constants (detached) so grads only hit backbone, stateless so better
        try:
            core = club_module.modules_to_save[getattr(club_module, "active_adapter", "default")]
        except (AttributeError, KeyError):
            core = getattr(club_module, "original_module", club_module)
        mapping = {**{n: p.detach() for n, p in core.named_parameters()},
                **{n: b for n, b in core.named_buffers()}}
        val = functional_call(core, mapping, (z, lbl))
        return lmda * torch.sqrt(val*val + 1e-6) # we care about magnitude of MI, not sign
    
    def _club_supervision(self, z, club_module, lbl):
        if club_module is None or lbl is None:
            return z.new_zeros(())
        try:
            core = club_module.modules_to_save[getattr(club_module, "active_adapter", "default")]
        except Exception:
            core = getattr(club_module, "original_module", club_module)
        p0 = next(core.parameters())
        z = z.to(dtype=p0.dtype, device=p0.device)

        assert any(p.requires_grad for p in core.parameters()), \
        "CLUB params are frozen; add modules_to_save or unfreeze."
        
        # z is detached
        return core.learning_loss(z.detach(), lbl)
    
    def get_model(self):
        return self.model

    def kl_to_uniform(self, logits: torch.Tensor) -> torch.Tensor:
        logits_f = logits.float()
        logp = F.log_softmax(logits_f, dim=-1)
        C = logits_f.size(-1)
        target = logits_f.new_full(logits_f.shape, 1.0 / C)
        kl = F.kl_div(logp, target, reduction='batchmean', log_target=False)
        return kl.to(logits.dtype)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        gender: Optional[torch.FloatTensor] = None,
        race_major: Optional[torch.FloatTensor] = None,
        handedness: Optional[torch.FloatTensor] = None,
        anchor_age_group: Optional[torch.FloatTensor] = None,
        no_ce_loss = False,
        dac_evaluation = False, # Set this to False for training,
        cur_lam: Optional[float] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # For fairness fine-tuning, we need access to intermediate layers
        if self.fairness_finetune:
            output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_mask = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        device = hidden_states.device
        # Calculate fairness loss BEFORE applying LM head
        fairness_loss = None
        
        # Evaluation mode for DAC, gives accuracies for dem classifiers. 
        if getattr(self, "dac_modules", None) is not None and dac_evaluation:
            assert outputs.hidden_states is not None, "Call with output_hidden_states=True for DAC eval."
            z = self._collect_layer_summaries(outputs.hidden_states, image_mask, self.tapped_layers)
            dem_logits = {dem: self.dac_modules[dem](z.detach()) for dem in self.dac_modules.keys()}
            batch_dem = {
                "gender": None if gender is None else gender.view(-1).long().to(device),
                "race": None if race_major is None else race_major.view(-1).long().to(device),
                "age": None if anchor_age_group is None else anchor_age_group.view(-1).long().to(device),
            }

            out_eval = {}
            if "gender" in dem_logits:
                lg = dem_logits["gender"]
                out_eval["gender"] = {"logits": lg, "pred": lg.argmax(dim=-1), "label": batch_dem["gender"]}
            if "race" in dem_logits:
                lg = dem_logits["race"]
                out_eval["race"] = {"logits": lg, "pred": lg.argmax(dim=-1), "label": batch_dem["race"]}
            if "age" in dem_logits:
                lg = dem_logits["age"]
                out_eval["age"] = {"logits": lg, "pred": lg.argmax(dim=-1), "label": batch_dem["age"]}
            return out_eval

        if self.fairness_finetune:
            assert self.training, "Fairness fine-tuning only in training mode."
            assert outputs.hidden_states is not None, "Enable output_hidden_states when fairness_finetune=True."
            batch_dem = {
                "gender": gender.view(-1).long().to(device) if "gender" in self.target_dem else None,
                "race": race_major.view(-1).long().to(device) if "race" in self.target_dem else None,
                "age": anchor_age_group.view(-1).long().to(device) if "age" in self.target_dem else None,
                "handedness": handedness.view(-1).long().to(device) if "handedness" in self.target_dem else None,
            }

            if self.debiasing_method == "dac_qkv":
                z = self._collect_layer_summaries(outputs.hidden_states, image_mask, self.tapped_layers) # [B, L_taps, D]
                dac_losses = {}
                dac_acc = {}
                for dem, dac_module in self.dac_modules.items():
                    y = batch_dem[dem].to(device)
                    assert y is not None, f"Batch must include {dem} labels."
                    
                    if self.fairness_stage == "debias_joint":
                        logits_dac = dac_module(z.detach())
                        loss_dac_classify = F.cross_entropy(logits_dac, y)  # Positive: maximize accuracy
                        logits_encoder = dac_module(z)  
                        loss_encoder_fool = -F.cross_entropy(logits_encoder, y)  # Negative: fool DAC
                        adv_dac_weight = getattr(self, 'adversarial_dac_weight', 1.0)
                        adv_enc_weight = getattr(self, 'adversarial_encoder_weight', 0.1)
                        
                        loss_dem = adv_dac_weight * loss_dac_classify + adv_enc_weight * loss_encoder_fool
                        if not hasattr(self, '_last_fairness_components'):
                            self._last_fairness_components = {}
                        self._last_fairness_components[f"dac_{dem}_classify_loss"] = loss_dac_classify.detach()
                        self._last_fairness_components[f"encoder_{dem}_fool_loss"] = loss_encoder_fool.detach()
                        
                        dac_losses[dem] = loss_dem
                        continue
                    
                    elif self.fairness_stage == "pretrain_dac": 
                        logits_a = dac_module(z.detach())  # [B, label_num], stop gradients to the decoder 
                        
                        # Calculate loss based on dac_loss_mode
                        if self.dac_loss_mode == "ce":
                            loss_dem = F.cross_entropy(logits_a, y)
                        elif self.dac_loss_mode == "focal":
                            loss_dem = focal_loss(logits_a, y, 
                                                alpha=self.focal_loss_alpha, 
                                                gamma=self.focal_loss_gamma)
                        elif self.dac_loss_mode == "ce_weighted":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            loss_dem = F.cross_entropy(logits_a, y, weight=class_weights)
                        elif self.dac_loss_mode == "ce_weighted_log_prior":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            log_priors = torch.log(priors).to(device)
                            loss_dem = F.cross_entropy(logits_a - self.tau * log_priors, y, weight=class_weights)
                        else:
                            raise ValueError(f"Unknown dac_loss_mode {self.dac_loss_mode}")
                        
                        # Calculate accuracy in eval mode (disable dropout for accurate measurement)
                        with torch.no_grad():
                            preds = logits_a.argmax(dim=-1)
                            acc = (preds == y).float().mean().item()

                        if torch.rand(1).item() <= 0.01:  # Print 1% of batches
                            print(f"\n[DAC {dem}]")
                            print(f"  Loss ({self.dac_loss_mode}): {loss_dem.item():.4f}")
                            print(f"  Batch Accuracy (eval mode): {acc:.3f}")
                            print(f"  Logits: {logits_a[0].tolist()}")  # First sample
                            print(f"  True labels: {y.tolist()}")
                            print(f"  Predictions: {preds.tolist()}")
                            print(f"  True dist: {torch.bincount(y, minlength=2)}")
                            print(f"  Pred dist: {torch.bincount(preds, minlength=2)}")
                    else:
                        # dac_module.train()
                        # for p in dac_module.parameters():
                        #     p.requires_grad_(True)
                        
                        z_adv = GradientReversalLayer.apply(z, cur_lam)
                        logits_a = dac_module(z_adv)  # [B, label_num]
                        if self.dac_loss_mode == "ce":
                            loss_dem = F.cross_entropy(logits_a, y)
                        elif self.dac_loss_mode == "focal":
                            loss_dem = focal_loss(logits_a, y, 
                                                alpha=self.focal_loss_alpha, 
                                                gamma=self.focal_loss_gamma)
                        elif self.dac_loss_mode == "ce_weighted":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            loss_dem = F.cross_entropy(logits_a, y, weight=class_weights)
                        elif self.dac_loss_mode == "ce_weighted_log_prior":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            log_priors = torch.log(priors).to(device)
                            loss_dem = F.cross_entropy(logits_a - self.tau * log_priors, y, weight=class_weights)
                        else:
                            raise ValueError(f"Unknown dac_loss_mode {self.dac_loss_mode}")
                        
                        # Track DAC accuracy for monitoring
                        if not hasattr(self, '_last_fairness_components'):
                            self._last_fairness_components = {}
                        with torch.no_grad():
                            preds = logits_a.argmax(dim=-1)
                            acc = (preds == y).float().mean().item()
                        self._last_fairness_components[f"dac_{dem}_acc"] = acc
                        
                        if torch.rand(1).item() <= 0.01:
                            print(f"\n[DAC {dem} - GRL Debias]")
                            print(f"  Loss (CE with GRL): {loss_dem.item():.4f}")
                            print(f"  Batch Accuracy: {acc:.3f}")
                            print(f"  Lambda: {cur_lam}")
                            print(f"  True dist: {torch.bincount(y, minlength=getattr(self, f'{dem}_num'))}")
                            print(f"  Pred dist: {torch.bincount(preds, minlength=getattr(self, f'{dem}_num'))}")

                    dac_losses[dem] = loss_dem*100 if self.fairness_stage == "pretrain_dac" else loss_dem
                    # dac_losses[dem] = loss_dem*100
                    dac_acc[dem] = acc

                if len(dac_losses) > 0:
                    fairness_loss = sum(dac_losses.values())
                
                self._last_fairness_components = {
                    **{f"dac_{dem}_loss": v.detach() for dem, v in dac_losses.items()},
                    **{f"dac_{dem}_acc": dac_acc[dem] for dem in dac_acc.keys()},
                }
            
            elif self.debiasing_method == "dac_mlp":
                # Same logic as dac_qkv but with MLP classifier
                z = self._collect_layer_summaries(outputs.hidden_states, image_mask, self.tapped_layers) # [B, L_taps, D]
                dac_losses = {}
                for dem, dac_module in self.dac_modules.items():
                    y = batch_dem[dem].to(device)
                    assert y is not None, f"Batch must include {dem} labels."
                    if self.fairness_stage == "pretrain_dac":
                        logits_a = dac_module(z.detach())  # [B, label_num]
                        
                        # Calculate loss based on dac_loss_mode
                        if self.dac_loss_mode == "ce":
                            loss_dem = F.cross_entropy(logits_a, y)
                        elif self.dac_loss_mode == "focal":
                            loss_dem = focal_loss(logits_a, y, 
                                                alpha=self.focal_loss_alpha, 
                                                gamma=self.focal_loss_gamma)
                        elif self.dac_loss_mode == "ce_weighted":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            loss_dem = F.cross_entropy(logits_a, y, weight=class_weights)
                        elif self.dac_loss_mode == "ce_weighted_log_prior":
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            log_priors = torch.log(priors).to(device)
                            loss_dem = F.cross_entropy(logits_a - self.tau * log_priors, y, weight=class_weights)
                        else:
                            raise ValueError(f"Unknown dac_loss_mode {self.dac_loss_mode}")
                        
                        with torch.no_grad():
                            preds = logits_a.argmax(dim=-1)
                            batch_acc = (preds == y).float().mean().item()
                        
                        if not hasattr(self, '_last_fairness_components'):
                            self._last_fairness_components = {}
                        self._last_fairness_components[f"dac_{dem}_acc"] = batch_acc

                        if torch.rand(1).item() <= 0.01:
                            print(f"\n[DAC-MLP {dem}]")
                            print(f"  Loss ({self.dac_loss_mode}): {loss_dem.item():.4f}")
                            print(f"  Batch Accuracy (eval mode): {batch_acc:.3f}")
                            print(f"  Logits: {logits_a[0].tolist()}")
                            print(f"  True labels: {y.tolist()}")
                            print(f"  Predictions: {preds.tolist()}")
            
                    else:
                        # GRL debias stage
                        # if not self.dac_frozen:
                        #     dac_module.train()
                        #     for p in dac_module.parameters():
                        #         p.requires_grad_(True)
                        z_adv = GradientReversalLayer.apply(z, cur_lam)
                        logits_a = dac_module(z_adv)
                        loss_dem = F.cross_entropy(logits_a, y)
                        
                        if not hasattr(self, '_last_fairness_components'):
                            self._last_fairness_components = {}
                        with torch.no_grad():
                            preds = logits_a.argmax(dim=-1)
                            acc = (preds == y).float().mean().item()
                        self._last_fairness_components[f"dac_{dem}_acc"] = acc
                        
                        if torch.rand(1).item() <= 0.01:
                            print(f"\n[DAC-MLP {dem} - GRL Debias]")
                            print(f"  Loss (CE with GRL): {loss_dem.item():.4f}")
                            print(f"  Batch Accuracy: {acc:.3f}")
                            print(f"  Lambda: {cur_lam}")

                    dac_losses[dem] = loss_dem*100 if self.fairness_stage == "pretrain_dac" else loss_dem

                if len(dac_losses) > 0:
                    fairness_loss = sum(dac_losses.values())
                
                self._last_fairness_components = {
                    **{f"dac_{dem}_loss": v.detach() for dem, v in dac_losses.items()},
                }
            elif self.debiasing_method == "mi_club":
                # Option 1: Use middle layer for better disentanglement
                if hasattr(self, 'hidden_depth') and self.hidden_depth < 1.0 and output_hidden_states and outputs.hidden_states is not None:
                    hidden_for_fairness = outputs.hidden_states[int(len(outputs.hidden_states) * self.hidden_depth)]
                else:
                    # Use final hidden states for fairness
                    hidden_for_fairness = hidden_states

                mi_mask = self._select_mi_mask(attention_mask=attention_mask, image_mask=image_mask)
                z = self._pool_hidden2(hidden_for_fairness, mi_mask)
                device = z.device 
                if self.fairness_stage == "pretrain_dac":
                    dac_losses = {}
                    for dem, club_module in self.club_modules.items():
                        if self.dac_loss_mode == "ce_weighted_log_prior":
                            y = batch_dem[dem].to(device)
                            logits_a = club_module.variational_net(z.detach())
                            class_weights, priors = class_balanced_weights(self.dem_class_counts[dem], device=device, dtype=logits_a.dtype)
                            log_priors = torch.log(priors).to(device)
                            loss_dem = F.cross_entropy(logits_a - self.tau * log_priors, y, weight=class_weights)
                        dac_losses[dem] = loss_dem * 100 
                        with torch.no_grad():
                            preds = logits_a.argmax(dim=-1)
                            acc = (preds == y).float().mean().item()
                        if torch.rand(1).item() <= 0.01:  # Print 1% of batches
                            print(f"\n[DAC {dem}]")
                            print(f"  Loss ({self.dac_loss_mode}): {loss_dem.item():.4f}")
                            print(f"  Batch Accuracy (eval mode): {acc:.3f}")
                            print(f"  Logits: {logits_a[0].tolist()}")  # First sample
                            print(f"  True labels: {y.tolist()}")
                            print(f"  Predictions: {preds.tolist()}")
                            print(f"  True dist: {torch.bincount(y, minlength=2)}")
                            print(f"  Pred dist: {torch.bincount(preds, minlength=2)}")

                    if len(dac_losses) > 0:
                        fairness_loss = sum(dac_losses.values())
                    self._last_fairness_components = {
                        **{f"club_{dem}_pretrain_loss": v.detach() for dem, v in dac_losses.items()},
                    }
                elif self.fairness_stage == "debias":
                    mi_losses, club_losses = {}, {}    
                    for dem, club_module in self.club_modules.items():
                        mi_losses[dem] = self._mi_term(z, club_module, getattr(self, f"{dem}_lmda"), batch_dem[dem])
                    for dem, club_module in self.club_modules.items(): # TODO uncomment this if commented, only added for ablation
                        club_losses[dem] = self._club_supervision(z, club_module, batch_dem[dem])
                    fairness_loss = sum(mi_losses.values()) + (sum(club_losses.values()) * self.mi_club_training_weight)

                    # Store components for logging
                    self._last_fairness_components = {
                        **{f"mi_{dem}_loss": v.detach() for dem, v in mi_losses.items()},
                        **{f"club_{dem}_loss": v.detach() for dem, v in club_losses.items()}
                    }
        # Apply LM head AFTER fairness computation
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and not no_ce_loss and self.fairness_stage != "pretrain_dac":
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_ce = loss_fct(shift_logits, shift_labels)
            loss = loss_ce * self.ce_loss_weight
            
        # Combine fairness and CE losses
        if fairness_loss is not None:
            fairness_scale = float(getattr(self, "fairness_lambda", 1.0))
            fairness_loss_scaled = fairness_loss * fairness_scale
            loss = loss + fairness_loss_scaled if loss is not None else fairness_loss_scaled
                        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            labels=labels,
        )
        if self.fairness_finetune and fairness_loss is not None:
            if not hasattr(out, 'losses'):
                out.losses = {}
            # Extract individual components for logging
            if hasattr(self, '_last_fairness_components'):
                components = self._last_fairness_components
                for k,v in components.items():
                    out.losses[k] = v
                if self.debiasing_method in ["dac_qkv", "dac_mlp"]:
                    out.losses["loss_dac_total"] = fairness_loss.detach()
                elif self.debiasing_method == "mi_club":
                    out.losses["loss_mi_total"] = fairness_loss.detach()
                out["losses"] = out.losses  # dict-style too
        if labels is not None and not no_ce_loss and self.fairness_stage != "pretrain_dac":
            if not hasattr(out, 'losses'):
                out.losses = {}
            out.losses["loss_ce"] = loss_ce
            # out["losses"] = out.losses  # update CE loss
        return out

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

# from transformers.models.auto.configuration_auto import CONFIG_MAPPING

# existing = CONFIG_MAPPING._mapping.get("llava") \
#         or CONFIG_MAPPING._extra_content.get("llava")

# if existing is None:
#     print("No existing 'llava' entry—safe to register.")
# elif existing is LlavaConfig:
#     print("'llava' is already registered to the same class; no need to re-register.")
# else:
#     print(f"'llava' maps to a different class ({existing}); will overwrite.")
    
AutoConfig.register("llava", LlavaConfig, exist_ok=True) #TODO check if exists ok is TRUE is correct choice here.
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)


# import traceback, sys, inspect
# lineno = inspect.currentframe().f_lineno + 2  # adjust if you want the line *after*
# print(f"\n=========================Registering llava at {__file__}:{lineno}==========================")
# traceback.print_stack(file=sys.stdout)

# # now do the actual registration (set exist_ok=True if you want to overwrite silently)
# AutoConfig.register("llava", LlavaConfig, exist_ok=False)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)