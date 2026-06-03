# -*- coding: utf-8 -*-
r"""
BERT4RecF
################################################
BERT4Rec with Feature side information.

Extends BERT4Rec by adding a FeatureSeqEmbLayer that looks up item
features (category IDs, etc.) and concatenates them with item embeddings
before feeding into the bidirectional Transformer.

Reference:
    BERT4Rec: Fei Sun et al. "BERT4Rec: Sequential Recommendation with
    Bidirectional Encoder Representations from Transformer." CIKM 2019.

    Feature injection follows the same pattern as SASRecF:
    feature_emb || item_emb → concat_layer → Transformer
"""

import torch
from torch import nn

from recbole.model.sequential_recommender.bert4rec import BERT4Rec
from recbole.model.layers import FeatureSeqEmbLayer
from recbole.utils import FeatureType


class BERT4RecF(BERT4Rec):
    """BERT4Rec with item feature side information.

    Item features (e.g. category IDs) are looked up via FeatureSeqEmbLayer
    and concatenated with item ID embeddings. A linear projection maps
    the combined representation back to hidden_size before entering
    the bidirectional Transformer.
    """

    def __init__(self, config, dataset):
        # ── BERT4Rec base init (without nn.Module.__init__ duplicate) ────
        # We call BERT4Rec.__init__ first, then override with feature layers
        super(BERT4RecF, self).__init__(config, dataset)

        # ── Feature embedding layer ──────────────────────────────────
        self.selected_features = config["selected_features"]
        self.pooling_mode = config["pooling_mode"]
        self.device = config["device"]

        # Count how many feature fields (each adds embedding_size dims)
        self.num_feature_field = sum(
            (
                1
                if dataset.field2type[field] != FeatureType.FLOAT_SEQ
                else dataset.num(field)
            )
            for field in config["selected_features"]
        )

        self.feature_embed_layer = FeatureSeqEmbLayer(
            dataset,
            self.hidden_size,
            self.selected_features,
            self.pooling_mode,
            self.device,
        )

        # Project concatenated [item_emb || feature_embs] back to hidden_size
        self.feat_concat_layer = nn.Linear(
            self.hidden_size * (1 + self.num_feature_field), self.hidden_size
        )

        # Re-register feature_embed_layer for checkpoint save/load
        self.other_parameter_name = getattr(self, "other_parameter_name", [])
        if isinstance(self.other_parameter_name, list):
            self.other_parameter_name.append("feature_embed_layer")

        # Re-initialize the new linear layer
        self.apply(self._init_weights)

    def forward(self, item_seq):
        """Forward pass with feature injection.

        Args:
            item_seq: [B, L] — may contain MASK token (value = self.mask_token = n_items)
                      at masked positions during training.

        Returns:
            output: [B, L, H]
        """
        # Track MASK positions BEFORE replacing with 0 for lookup safety.
        # We need the mask later to zero out features at those positions —
        # MASK tokens should carry no item-specific feature signal, only
        # the learnable MASK embedding in item_emb.
        mask_positions = (item_seq == self.mask_token)  # [B, L]
        feat_seq = item_seq.clone()
        feat_seq[mask_positions] = 0  # safe lookup: MASK token (n_items) → pad (0)

        # Item ID embeddings (preserves MASK token embedding at masked positions)
        item_emb = self.item_embedding(item_seq)  # [B, L, H]

        # Feature embeddings from FeatureSeqEmbLayer
        sparse_embedding, dense_embedding = self.feature_embed_layer(None, feat_seq)
        sparse_embedding = sparse_embedding["item"]
        dense_embedding = dense_embedding["item"]

        # Concatenate sparse and dense feature embeddings
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
        )  # [B, L, num_feat*H]

        # Zero out features at masked positions:
        #   MASK token's item_emb already signals "predict me";
        #   injecting item 0's real category features here would
        #   create a spurious training signal.
        feature_emb[mask_positions.unsqueeze(-1).expand_as(feature_emb)] = 0.0

        # Concatenate item + feature embeddings and project
        input_concat = torch.cat(
            (item_emb, feature_emb), -1
        )  # [B, L, H * (1 + num_feat)]
        input_emb = self.feat_concat_layer(input_concat)  # [B, L, H]

        # Position embeddings
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = input_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        # Bidirectional attention + FFN output (same as BERT4Rec)
        extended_attention_mask = self.get_attention_mask(
            item_seq, bidirectional=True
        )
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        ffn_output = self.output_ffn(trm_output[-1])
        ffn_output = self.output_gelu(ffn_output)
        output = self.output_ln(ffn_output)
        return output  # [B, L, H]
