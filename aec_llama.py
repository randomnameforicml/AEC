from transformers.models.llama.modeling_llama import (
    repeat_kv,
    LlamaAttention,
    LlamaModel,
    apply_rotary_pos_emb
)

from transformers.modeling_outputs import BaseModelOutputWithPast
import math
import types
import torch
from torch import nn
from flash_attn import flash_attn_func
from typing import List, Optional, Tuple, Union
from functools import partial
import json

# Global dictionary to store the offline profile loaded from JSON
AEC_PROFILE = {}

def load_aec_profile(profile_path):
    global AEC_PROFILE
    with open(profile_path, 'r') as f:
        AEC_PROFILE = json.load(f)
        
def compute_entropy(probs):
    """Compute entropy along the last dimension."""
    eps = 1e-9
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

def aec_llama_attention_prefill_query(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    assert past_key_value is not None

    if len(past_key_value) == 4:
        past_key, past_value, past_position = past_key_value[0], past_key_value[1], past_key_value[2]
        current_position = past_position.max().item() + 1
        self.len_context = past_key.shape[2] - self.len_prefix
    else:
        (past_key, past_value, past_position) = past_key_value
        current_position = past_position.max().item() + 1

    key_position_ids = position_ids - position_ids.min().item() + current_position

    cos, sin = self.rotary_emb(value_states, key_position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, key_position_ids)

    key_states = torch.cat([past_key, key_states], dim=2)
    value_states = torch.cat([past_value, value_states], dim=2)
    position_states = torch.cat([past_position, key_position_ids], dim=-1)
    
    past_key_value = (key_states, value_states, position_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    key_states_context = key_states[:, :, self.len_prefix:self.len_prefix+self.len_context]
    key_states_other = torch.cat([key_states[:, :, :self.len_prefix], key_states[:, :, self.len_prefix+self.len_context:]], dim=-2)
    value_states_context = value_states[:, :, self.len_prefix:self.len_prefix+self.len_context]
    value_states_other = torch.cat([value_states[:, :, :self.len_prefix], value_states[:, :, self.len_prefix+self.len_context:]], dim=-2)

    layer_idx_str = str(self.layer_idx)
    target_entropy = AEC_PROFILE.get(layer_idx_str, {}).get('target_entropy', 1.0)
    target_lse = AEC_PROFILE.get(layer_idx_str, {}).get('target_lse', 1.0)

    # 1. Base attention scores from flash attention (return_attn_probs doesn't return raw scores, so we use math or recompute)
    # However, flash_attn_func can return lse.
    # We will use pure pytorch for the alignment calculations of the last query token to keep it clean, 
    # since alignment only really matters for the queries.
    # For APE, q_len during prefill_query is usually small.
    
    # Calculate unscaled logits for context and other
    scale_factor = 1 / math.sqrt(self.head_dim)
    # We compute raw attention scores for context and other
    raw_scores_context = torch.matmul(query_states, key_states_context.transpose(2, 3)) * scale_factor # (bsz, heads, q_len, ctx_len)
    raw_scores_other = torch.matmul(query_states, key_states_other.transpose(2, 3)) * scale_factor # (bsz, heads, q_len, oth_len)
    
    # Also we need to apply causal mask to 'other' if q_len > 1
    if q_len > 1:
        # standard causal mask logic
        causal_mask = torch.full((q_len, key_states_other.shape[3]), float("-inf"), device=query_states.device, dtype=query_states.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=key_states_other.shape[3] - q_len + 1)
        raw_scores_other = raw_scores_other + causal_mask.unsqueeze(0).unsqueeze(0)

    max_score_context = torch.max(raw_scores_context, dim=-1, keepdim=True)[0]
    max_score_other = torch.max(raw_scores_other, dim=-1, keepdim=True)[0]

    # --- AEC Step 2: Online Calibration (Binary Search for t) ---
    t_min = 0.1
    t_max = 5.0
    t_opt = 1.0
    
    for _ in range(7): # 7 steps of binary search is usually enough
        t_mid = (t_min + t_max) / 2
        
        # Apply temperature
        scaled_score_ctx = raw_scores_context / t_mid
        scaled_score_oth = raw_scores_other # Note: APE original code applies T only to context flash_attn. Wait, APE scales lse_context.
        # Let's apply t to everything to properly compute combined entropy based on APE's logic.
        # Wait, looking at APE original:
        # lse_context = flash_attn(..., softmax_scale = base_scale / temperature) -> this means T is applied to context.
        # other is normal.
        
        # We follow APE's assumption: scale context logits
        # Calculate probs to find entropy
        lse_ctx_t = torch.logsumexp(scaled_score_ctx, dim=-1, keepdim=True)
        lse_oth = torch.logsumexp(raw_scores_other, dim=-1, keepdim=True)
        
        combined_lse = torch.logaddexp(lse_ctx_t, lse_oth)
        
        p_ctx = torch.exp(scaled_score_ctx - combined_lse)
        p_oth = torch.exp(raw_scores_other - combined_lse)
        
        ent_ctx = compute_entropy(p_ctx)
        ent_oth = compute_entropy(p_oth)
        current_entropy = ent_ctx + ent_oth # Total entropy
        
        mean_entropy = current_entropy.mean().item()
        
        if mean_entropy > target_entropy:
            # Entropy too high -> distribution too flat -> decrease temperature to sharpen
            t_max = t_mid
        else:
            t_min = t_mid
            
    t_opt = (t_min + t_max) / 2
    
    # --- AEC Step 3: Online Scale Calculation (Analytic s) ---
    # With optimal t, calculate LSE
    scaled_score_ctx = raw_scores_context / t_opt
    lse_ctx_t = torch.logsumexp(scaled_score_ctx, dim=-1, keepdim=True)
    lse_oth = torch.logsumexp(raw_scores_other, dim=-1, keepdim=True)
    
    # We want to find s such that: logsumexp(s * lse_ctx_t + lse_oth) = target_lse
    # In APE: lse_context = lse_context * (scale * temperature).
    # This means they scale the LSE directly linearly: lse_context_scaled = s * t * lse_context.
    # Let's solve for S:
    # Target LSE = logaddexp(s * lse_ctx_t, lse_oth) (Approximate)
    # Analytically to match exactly:
    
    mean_lse_oth = lse_oth.mean().item()
    mean_lse_ctx_t = lse_ctx_t.mean().item()
    
    # We need: log(exp(S * mean_lse_ctx) + exp(mean_lse_oth)) = target_lse
    # exp(S * mean_lse_ctx) = exp(target_lse) - exp(mean_lse_oth)
    
    exp_target = math.exp(target_lse)
    exp_oth = math.exp(mean_lse_oth)
    
    if exp_target > exp_oth:
        s_opt = math.log(exp_target - exp_oth) / (mean_lse_ctx_t + 1e-9)
    else:
        # Fallback if target is smaller than the unscaled 'other' part
        s_opt = 1.0

    # Finally apply them properly to flash attention calls to get the fast outputs
    # Notice APE scales lse_context = lse_context * (scale * temperature)
    # which is exactly what we derived.
    
    attn_output_context, lse_context, _ = flash_attn_func(
        query_states.transpose(1, 2), 
        key_states_context.transpose(1, 2), 
        value_states_context.transpose(1, 2), 
        causal=False, 
        softmax_scale = 1 / (math.sqrt(self.head_dim) * t_opt)
    )
    
    attn_output_other, lse_other, _ = flash_attn_func(
        query_states.transpose(1, 2), 
        key_states_other.transpose(1, 2), 
        value_states_other.transpose(1, 2), 
        causal=True
    )
    
    lse_context = lse_context.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)
    lse_other = lse_other.transpose(1, 2).unsqueeze(-1).to(query_states.dtype)

    # Apply our analytic scale
    # In APE's formulation, scale translates linearly to lse_context
    lse_context = lse_context * (s_opt * t_opt)

    attn_weights = torch.cat([lse_context, lse_other], dim=-1).unsqueeze(dim=-2)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    
    value_states = torch.cat([attn_output_context.unsqueeze(-2), attn_output_other.unsqueeze(-2)], dim=-2)
    attn_output = torch.matmul(attn_weights, value_states).squeeze(dim=-2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def enable_aec_llama_attention_prefill_query(model, profile_path):
    load_aec_profile(profile_path)
    
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_aec_llama_attention_prefill_query(
                module, profile_path
            )

        if isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                aec_llama_attention_prefill_query, model._modules[name]
            )
