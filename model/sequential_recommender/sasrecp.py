# -*- coding: utf-8 -*-
"""
SASRecP — SASRec with three plug-and-play innovations:

1. RoPE (Rotary Position Embedding) — Su et al. 2021
   Replaces absolute position embeddings with rotary encoding so attention
   depends on relative position distance (j-i), not absolute index.

2. Time Interval Attention Bias
   Adds a learnable bias to attention scores based on real timestamps:
   items closer in time → higher attention weight.

3. Feature-aware Contrastive Loss (InfoNCE)
   Auxiliary loss: items sharing the same category label are pulled closer
   in a normalized embedding space. Uses hierarchical category weights.

All three are controlled by boolean flags in the config for clean ablation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.sasrecf import SASRecF
from recbole.model.layers import (
    RotaryPositionEmbedding,
    compute_time_bias,
)
from recbole.model.loss import BPRLoss


class SASRecP(SASRecF):
    """SASRec with RoPE, TimeIntervalBias, and Contrastive auxiliary loss."""

    def __init__(self, config, dataset):
        # ── Let SASRecF set up item_emb, feature_embed_layer, concat_layer,
        #    TrmEncoder, LayerNorm, dropout, loss_fct, position_embedding ────
        super().__init__(config, dataset)

        # ── Innovation flags ───────────────────────────────────────────────
        self.use_rope = (config["use_rope"] if "use_rope" in config else False)
        self.use_time_bias = (config["use_time_bias"] if "use_time_bias" in config else False)
        self.time_bias_method = (config["time_bias_method"] if "time_bias_method" in config else "log")
        self.use_contra_loss = (config["use_contra_loss"] if "use_contra_loss" in config else False)
        self.contra_lambda = (config["contra_lambda"] if "contra_lambda" in config else 0.1)
        self.contra_temperature = (config["contra_temperature"] if "contra_temperature" in config else 0.07)
        self.contra_weighting = (config["contra_weighting"] if "contra_weighting" in config else "binary")
        self.contra_category_field = (config["contra_category_field"] if "contra_category_field" in config else "second_level_category_id")

        # ── RoPE: replace absolute position embedding ──────────────────────
        if self.use_rope:
            if hasattr(self, "position_embedding"):
                del self.position_embedding
            self.attention_head_size = int(self.hidden_size / self.n_heads)
            cos, sin = RotaryPositionEmbedding.precompute(
                self.max_seq_length, self.attention_head_size, device="cpu"
            )
            self.register_buffer("rope_cos_table", cos)  # [max_seq_length, head_dim]
            self.register_buffer("rope_sin_table", sin)

        # ── Time bias: learnable scale factor ──────────────────────────────
        if self.use_time_bias:
            self.time_bias_weight = nn.Parameter(torch.tensor(0.01))
            self.TIME_SEQ_FIELD = config["TIME_FIELD"] + config["LIST_SUFFIX"]

        # ── Contrastive loss: category data ────────────────────────────────
        if self.use_contra_loss:
            item_feat = dataset.get_item_feature()
            cat_data = item_feat[self.contra_category_field]
            cat_values = np.asarray(cat_data, dtype=int)
            self.register_buffer(
                "item_category_ids",
                torch.tensor(cat_values, dtype=torch.long),
            )
            # For hierarchical weighting: also load coarser categories
            self.contra_L1_field = (config["contra_L1_field"] if "contra_L1_field" in config else "first_level_category_id")
            self.contra_L3_field = (config["contra_L3_field"] if "contra_L3_field" in config else "third_level_category_id")
            if self.contra_weighting == "hierarchical":
                l1_data = item_feat[self.contra_L1_field]
                self.register_buffer(
                    "item_L1_ids",
                    torch.tensor(np.asarray(l1_data, dtype=int), dtype=torch.long),
                )
                l3_data = item_feat[self.contra_L3_field]
                self.register_buffer(
                    "item_L3_ids",
                    torch.tensor(np.asarray(l3_data, dtype=int), dtype=torch.long),
                )

    # ── forward ───────────────────────────────────────────────────────────

    def forward(self, item_seq, item_seq_len, timestamps=None):
        # 1. Item + feature embeddings (same as SASRecF)
        item_emb = self.item_embedding(item_seq)

        sparse_embedding, dense_embedding = self.feature_embed_layer(None, item_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]
        feature_table = []
        if sparse_embedding is not None:
            feature_table.append(sparse_embedding)
        if dense_embedding is not None:
            feature_table.append(dense_embedding)
        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        feat_num, embedding_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(
            table_shape[:-2] + (feat_num * embedding_size,)
        )
        input_concat = torch.cat((item_emb, feature_emb), -1)
        input_emb = self.concat_layer(input_concat)  # [B, L, H]

        # 2. Position encoding
        extra_kwargs = {}
        if not self.use_rope:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=item_seq.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = self.position_embedding(position_ids)
            input_emb = input_emb + position_embedding
        else:
            B, L = item_seq.shape
            cos = self.rope_cos_table[:L].unsqueeze(0).expand(B, -1, -1)
            sin = self.rope_sin_table[:L].unsqueeze(0).expand(B, -1, -1)
            extra_kwargs["rope_cos"] = cos  # [B, L, head_dim]
            extra_kwargs["rope_sin"] = sin

        # 3. Time interval bias
        if self.use_time_bias and timestamps is not None:
            time_bias = compute_time_bias(
                timestamps,
                method=self.time_bias_method,
                weight=self.time_bias_weight,
            )
            extra_kwargs["time_bias"] = time_bias  # [B, 1, L, L]

        # 4. Transformer (same as SASRecF + extra_kwargs)
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
            **extra_kwargs,
        )
        output = trm_output[-1]
        seq_output = self.gather_indexes(output, item_seq_len - 1)
        return seq_output  # [B, H]

    # ── calculate_loss ────────────────────────────────────────────────────

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # Timestamps for time bias
        timestamps = None
        if self.use_time_bias:
            if hasattr(self, "TIME_SEQ_FIELD") and self.TIME_SEQ_FIELD in interaction:
                timestamps = interaction[self.TIME_SEQ_FIELD]

        seq_output = self.forward(item_seq, item_seq_len, timestamps=timestamps)

        # ── Main loss (same as SASRecF) ────────────────────────────────
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            main_loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            main_loss = self.loss_fct(logits, pos_items)

        # ── Contrastive auxiliary loss ─────────────────────────────────
        if self.use_contra_loss:
            contra_loss = self._contrastive_loss(item_seq)
            return main_loss, contra_loss

        return main_loss

    # ── Contrastive loss ──────────────────────────────────────────────────

    def _contrastive_loss(self, item_seq):
        """InfoNCE loss over item embeddings, pulling same-category items closer.

        Supports two weighting modes:
          - "binary":     same L2 category = positive (weight=1)
          - "hierarchical": same L3=1.0, same L2 only=0.5, same L1 only=0.2
        """
        # Collect unique non-padding items from this batch
        items = torch.unique(item_seq[item_seq > 0])
        if items.size(0) < 2:
            return torch.tensor(0.0, device=item_seq.device, requires_grad=True)

        cats = self.item_category_ids[items]  # [N] — L2 category IDs
        embs = self.item_embedding(items)  # [N, H]
        embs = F.normalize(embs, dim=-1)

        sim = (embs @ embs.T) / self.contra_temperature  # [N, N]

        if self.contra_weighting == "binary":
            # Same L2 category → positive
            pos_mask = (cats.unsqueeze(0) == cats.unsqueeze(1)) & ~torch.eye(
                items.size(0), dtype=torch.bool, device=item_seq.device
            )
        else:  # "hierarchical"
            l1 = self.item_L1_ids[items]
            l3 = self.item_L3_ids[items]
            same_l1 = l1.unsqueeze(0) == l1.unsqueeze(1)
            same_l2 = cats.unsqueeze(0) == cats.unsqueeze(1)
            same_l3 = l3.unsqueeze(0) == l3.unsqueeze(1)
            not_self = ~torch.eye(
                items.size(0), dtype=torch.bool, device=item_seq.device
            )
            # Hierarchy: L3 > L2 > L1
            pos_mask = torch.zeros(items.size(0), items.size(0), device=item_seq.device)
            pos_mask = pos_mask.masked_fill(same_l3 & not_self, 1.0)
            pos_mask = pos_mask.masked_fill(
                same_l2 & ~same_l3 & not_self, 0.5
            )
            pos_mask = pos_mask.masked_fill(
                same_l1 & ~same_l2 & not_self, 0.2
            )

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=item_seq.device, requires_grad=True)

        exp_sim = torch.exp(sim)  # [N, N]

        if self.contra_weighting == "binary":
            numerator = (exp_sim * pos_mask.float()).sum(dim=1)  # [N]
        else:
            # Soft weighting: weight * exp(sim) for positives
            numerator = (exp_sim * pos_mask).sum(dim=1)  # [N]

        denominator = exp_sim.sum(dim=1)  # [N]
        has_pos = pos_mask.any(dim=1)

        loss = -torch.log(numerator[has_pos] / denominator[has_pos] + 1e-8).mean()
        return loss * self.contra_lambda
