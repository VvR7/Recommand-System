#!/usr/bin/env python
"""
TEMPORAL leave-one-out evaluation — the correct protocol for sequential rec.

Problem with current evaluation:
  NeuMF/ItemKNN/BPR use ALL items as history INCLUDING evaluation targets.
  This is data leakage → inflated scores (GAUC 0.82 for NeuMF).
  BERT4Rec truncates to last 200 items → no leakage on early items → lower scores.

Fix: For each user, sort items by timestamp, split into "history" (first 80%)
  and "eval" (last 20%). Build user profile from history, score on eval items.
  No item appears in both history and eval.

This is usable for ALL methods (BERT4Rec, NeuMF, BPR, ItemKNN) and ensures
a truly fair comparison.

Usage:
  python evaluate_temporal.py --checkpoint saved/NeuMF-xxx.pth --model NeuMF --split test
  python evaluate_temporal.py --checkpoint result/BERT4Rec-xxx.pth --model BERT4Rec --split test
"""

import argparse
import os
import pickle
import sys
import warnings
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

# ── Constants ───────────────────────────────────────────────────────────────
POS_THRESHOLD = 0.7
NEG_THRESHOLD = 0.3
NUM_NEG = 99
SEED = 42
K_VALUES = [5, 10, 20]
HISTORY_RATIO = 0.8  # First 80% of items as history, last 20% as evaluation


def load_model_and_data(checkpoint_path, device):
    """Load trained model, config, and dataset from checkpoint."""
    from recbole.quick_start import load_data_and_model
    config, model, dataset, _, _, _ = load_data_and_model(checkpoint_path)
    model = model.to(device)
    model.eval()
    return config, model, dataset


def build_temporal_splits(test_data, vid2int):
    """For each user: sort items by timestamp, split into history/eval.

    Returns:
      user_histories: {uid: [internal_item_ids]} — first 80% items
      user_evals: {uid: {positives: [iids], hard_negs: [iids]}} — last 20%
    """
    pos = test_data[test_data["watch_ratio"] >= POS_THRESHOLD].copy()

    # Map to internal IDs
    pos["iid"] = pos["video_id"].astype(str).map(vid2int)
    pos = pos[pos["iid"].notna() & (pos["iid"] != 0)]
    pos["iid"] = pos["iid"].astype(int)

    # Split into items with valid timestamp (can be sorted) and NaN
    has_ts = pos[pos["timestamp"].notna()]
    no_ts = pos[pos["timestamp"].isna()]

    user_histories = {}
    user_evals = {}

    user_timestamps = {}  # {uid: [timestamps]} — for time-bias scoring

    for uid, grp in has_ts.groupby("user_id", sort=False):
        # Sort by timestamp
        grp = grp.sort_values("timestamp")
        items = grp["iid"].tolist()
        timestamps = grp["timestamp"].tolist()
        n = len(items)
        split = max(1, int(n * HISTORY_RATIO))  # At least 1 eval item
        user_histories[uid] = items[:split]
        user_timestamps[uid] = timestamps[:split]
        eval_items = items[split:]
        if eval_items:
            user_evals[uid] = {"positives": eval_items, "hard_negs": []}

    # NaN-timestamp items: append to history (no temporal order available)
    for uid, grp in no_ts.groupby("user_id", sort=False):
        items = grp["iid"].tolist()
        if uid in user_histories:
            user_histories[uid].extend(items)
        else:
            user_histories[uid] = items
        # No timestamps to add for these items — time bias will ignore them

    # Build hard negatives from ALL items with wr < NEG_THRESHOLD
    neg = test_data[test_data["watch_ratio"] < NEG_THRESHOLD].copy()
    neg["iid"] = neg["video_id"].astype(str).map(vid2int)
    neg = neg[neg["iid"].notna() & (neg["iid"] != 0)]
    neg["iid"] = neg["iid"].astype(int)

    for uid, grp in neg.groupby("user_id", sort=False):
        if uid in user_evals:
            user_evals[uid]["hard_negs"] = grp["iid"].tolist()

    # Filter: need at least 1 positive and NUM_NEG negatives
    valid_evals = {}
    for uid, ev in user_evals.items():
        if ev["positives"] and len(ev["hard_negs"]) >= NUM_NEG:
            valid_evals[uid] = ev

    return user_histories, user_timestamps, valid_evals


# ── Model-specific scoring builders ────────────────────────────────────────

def build_bert4rec_scorer(model, n_items, config, device):
    """BERT4Rec: mean-pool over ALL positions (no MASK, no reconstruct)."""
    MAX_LEN = config["MAX_ITEM_LIST_LENGTH"]
    ITEM_SEQ_FIELD = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    ITEM_LEN_FIELD = config["ITEM_LIST_LENGTH_FIELD"]
    item_emb = model.item_embedding.weight.data
    output_bias = model.output_bias.data

    from recbole.data.interaction import Interaction

    @torch.no_grad()
    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)

        seq_input = valid[-MAX_LEN:]
        seq_len = len(seq_input)
        padded = [0] * (MAX_LEN - seq_len) + seq_input

        inter = Interaction({
            ITEM_SEQ_FIELD: torch.tensor([padded], dtype=torch.long, device=device),
            ITEM_LEN_FIELD: torch.tensor([seq_len], dtype=torch.long, device=device),
        })

        # Forward WITHOUT reconstruct_test_data (no MASK token)
        # For BERT4RecP: model.forward() handles RoPE internally
        seq_output = model.forward(inter[ITEM_SEQ_FIELD])  # [1, L, H]

        # Mean-pool over non-padding positions
        mask = (inter[ITEM_SEQ_FIELD] != 0).float().unsqueeze(-1)
        pooled = (seq_output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        test_emb = item_emb[:n_items]
        bias = output_bias[:n_items] if output_bias.shape[0] > n_items else output_bias
        scores = torch.matmul(pooled, test_emb.transpose(0, 1))
        if bias.shape[0] == n_items:
            scores = scores + bias.unsqueeze(0)
        return scores.squeeze(0).cpu().numpy()

    return score


def build_neumf_scorer(model, n_items, device):
    """NeuMF: GMF+MLP virtual user embedding."""
    item_mf = model.item_mf_embedding.weight.data
    item_mlp = model.item_mlp_embedding.weight.data

    @torch.no_grad()
    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)
        valid_t = torch.tensor(valid, device=device, dtype=torch.long)
        user_mf = item_mf[valid_t].mean(dim=0)
        user_mlp = item_mlp[valid_t].mean(dim=0)
        mf_out = user_mf.unsqueeze(0) * item_mf
        user_mlp_exp = user_mlp.unsqueeze(0).expand(n_items, -1)
        mlp_in = torch.cat([user_mlp_exp, item_mlp], dim=1)
        mlp_out = model.mlp_layers(mlp_in)
        combined = torch.cat([mf_out, mlp_out], dim=1)
        scores = model.predict_layer(combined).squeeze(-1)
        return scores.cpu().numpy()

    return score


def build_bpr_scorer(model, n_items, device):
    """BPR: dot product with mean item embedding."""
    item_emb = model.item_embedding.weight.detach().cpu().numpy()

    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)
        user_vec = np.mean(item_emb[valid], axis=0)
        return user_vec @ item_emb.T

    return score


def build_itemknn_scorer(model, n_items, device):
    """ItemKNN: aggregate item-item similarities."""
    if hasattr(model.w, "tocsr"):
        W = model.w.tocsr()
    else:
        W = model.w

    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)
        if hasattr(W, "toarray"):
            agg = np.mean(W[valid].toarray(), axis=0).flatten()
        else:
            agg = np.mean(W[valid], axis=0)
            if hasattr(agg, "A1"):
                agg = agg.A1
        return agg

    return score


def build_deepfm_scorer(model, n_items, dataset, device):
    """DeepFM (ID-only): FM + MLP with virtual user embedding (mean of history items).

    This is the ID-only scorer for backward compatibility.
    For feature-rich DeepFM, use build_deepfm_feat_scorer.
    """
    return _build_deepfm_scorer_impl(model, n_items, dataset, device, feat_mode=False)


def build_deepfm_feat_scorer(model, n_items, dataset, device):
    """DeepFM (feature-rich): handles arbitrary token fields.

    Generalizes the ID-only scorer to support any number of feature fields
    (user features + item features). For cross-user eval:
    - user_id: replaced with mean of history item embeddings (virtual user)
    - user features (not user_id): filled with zero embeddings
    - video_id: candidate item's embedding
    - item features: candidate item's feature embeddings
    """
    return _build_deepfm_scorer_impl(model, n_items, dataset, device, feat_mode=True)


def _build_deepfm_scorer_impl(model, n_items, dataset, device, feat_mode=False):
    """Unified DeepFM scorer supporting both ID-only and feature-rich modes."""
    from recbole.utils import FeatureSource

    token_names = model.token_field_names
    token_dims = model.token_field_dims
    token_offsets = model.token_field_offsets
    token_weight = model.token_embedding_table.embedding.weight

    fo_weight = model.first_order_linear.token_embedding_table.embedding.weight

    # Identify field roles
    item_id_idx = token_names.index("video_id")
    item_id_offset = token_offsets[item_id_idx]
    item_id_dim = token_dims[item_id_idx]

    # Classify each field
    field_roles = []
    for i, name in enumerate(token_names):
        src = dataset.field2source.get(name)
        if name == "user_id":
            field_roles.append("user_id")
        elif name == "video_id":
            field_roles.append("item_id")
        elif src == FeatureSource.USER:
            field_roles.append("user_feat")
        elif src == FeatureSource.ITEM:
            field_roles.append("item_feat")
        elif src == FeatureSource.USER_ID:
            field_roles.append("user_id")
        elif src == FeatureSource.ITEM_ID:
            field_roles.append("item_id")
        else:
            field_roles.append("item_feat")

    # Pre-fetch item feature values for item_feat fields
    item_feat = dataset.get_item_feature()
    item_feat_vals = {}  # field_name -> tensor [n_items] of feature values
    for field_i, name in enumerate(token_names):
        if field_roles[field_i] == "item_feat":
            vals = item_feat.interaction[name][:n_items].to(device)
            item_feat_vals[name] = vals

    n_fields = len(token_names)
    E = model.embedding_size

    @torch.no_grad()
    def score(history_ids):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)

        valid_t = torch.tensor(valid, device=device, dtype=torch.long)

        field_embs = []
        first_order_parts = []

        for field_i in range(n_fields):
            offset = token_offsets[field_i]
            dim = token_dims[field_i]
            fw = token_weight[offset:offset + dim]      # [dim, E]
            fo_fw = fo_weight[offset:offset + dim, 0]    # [dim]
            role = field_roles[field_i]
            name = token_names[field_i]

            if role == "item_id":
                # Candidate items' ID embeddings: direct index
                field_embs.append(fw[:n_items])           # [I, E]
                first_order_parts.append(fo_fw[:n_items])  # [I]

            elif role == "item_feat":
                # Candidate items' feature embeddings: lookup by feature value
                feat_vals = item_feat_vals[name]          # [I]
                field_embs.append(fw[feat_vals])           # [I, E]
                first_order_parts.append(fo_fw[feat_vals]) # [I]

            elif role == "user_id":
                # Virtual user: mean of history item ID embeddings
                item_id_w = token_weight[item_id_offset:item_id_offset + item_id_dim]
                virtual = item_id_w[valid_t].mean(dim=0)   # [E]
                field_embs.append(virtual.unsqueeze(0).expand(n_items, -1))
                # First order: mean of history item biases as virtual user bias
                item_fo = fo_weight[item_id_offset:item_id_offset + item_id_dim, 0]
                virtual_fo = item_fo[valid_t].mean()
                first_order_parts.append(torch.full((n_items,), virtual_fo, device=device))

            elif role == "user_feat":
                # User features unavailable in cross-user → zero
                field_embs.append(torch.zeros(n_items, E, device=device))
                first_order_parts.append(torch.zeros(n_items, device=device))

        all_emb = torch.stack(field_embs, dim=1)     # [I, n_fields, E]
        first_order = torch.stack(first_order_parts, dim=1).sum(dim=1)  # [I]

        # FM
        fm_out = model.fm(all_emb).squeeze(-1)        # [I]

        # MLP
        mlp_in = all_emb.view(n_items, n_fields * E)  # [I, n_fields*E]
        mlp_out = model.mlp_layers(mlp_in)            # [I, H_last]
        deep_out = model.deep_predict_layer(mlp_out).squeeze(-1)  # [I]

        scores = first_order + fm_out + deep_out
        return scores.cpu().numpy()

    return score


def build_sasrecf_scorer(model, n_items, config, device):
    """SASRecF: SASRec + item features, mean-pool over positions."""
    MAX_LEN = config["MAX_ITEM_LIST_LENGTH"]
    ITEM_SEQ_FIELD = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    ITEM_LEN_FIELD = config["ITEM_LIST_LENGTH_FIELD"]

    from recbole.data.interaction import Interaction
    has_features = hasattr(model, "feature_embed_layer")

    @torch.no_grad()
    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)

        seq_input = valid[-MAX_LEN:]
        seq_len = len(seq_input)
        padded = [0] * (MAX_LEN - seq_len) + seq_input

        inter = Interaction({
            ITEM_SEQ_FIELD: torch.tensor([padded], dtype=torch.long, device=device),
            ITEM_LEN_FIELD: torch.tensor([seq_len], dtype=torch.long, device=device),
        })

        item_seq = inter[ITEM_SEQ_FIELD]  # [1, L]

        # Item embeddings
        item_emb = model.item_embedding(item_seq)  # [1, L, H]

        # Feature injection (SASRecF path)
        if has_features:
            sparse_emb, dense_emb = model.feature_embed_layer(None, item_seq)
            sparse_emb = sparse_emb["item"]
            dense_emb = dense_emb["item"]
            feature_table = []
            if sparse_emb is not None:
                feature_table.append(sparse_emb)
            if dense_emb is not None:
                feature_table.append(dense_emb)
            feature_table = torch.cat(feature_table, dim=-2)
            table_shape = feature_table.shape
            feat_num, embed_size = table_shape[-2], table_shape[-1]
            feature_emb = feature_table.view(
                table_shape[:-2] + (feat_num * embed_size,)
            )
            input_concat = torch.cat((item_emb, feature_emb), -1)
            # SASRecF uses concat_layer; BERT4RecF uses feat_concat_layer
            concat_layer = getattr(model, "feat_concat_layer", None) or getattr(model, "concat_layer", None)
            input_emb = concat_layer(input_concat)
        else:
            input_emb = item_emb

        # Position embeddings
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = model.position_embedding(position_ids)

        input_emb = input_emb + position_embedding
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)

        # Causal attention mask
        extended_attention_mask = model.get_attention_mask(item_seq)

        # Transformer encoder
        trm_output = model.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]  # [1, L, H]

        # Mean-pool over non-padding positions
        mask = (item_seq != 0).float().unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Dot-product with all item embeddings
        test_items_emb = model.item_embedding.weight[:n_items]
        scores = torch.matmul(pooled, test_items_emb.transpose(0, 1))
        return scores.squeeze(0).cpu().numpy()

    return score


def build_sasrecp_scorer(model, n_items, config, device):
    """SASRecP: SASRecF + RoPE + time_bias in evaluation.

    Replicates SASRecF forward logic with mean-pooling, adding RoPE
    injection into attention and time-interval bias from timestamps.
    """
    MAX_LEN = config["MAX_ITEM_LIST_LENGTH"]
    ITEM_SEQ_FIELD = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    ITEM_LEN_FIELD = config["ITEM_LIST_LENGTH_FIELD"]

    from recbole.data.interaction import Interaction
    from recbole.model.layers import compute_time_bias

    use_rope = getattr(model, "use_rope", False)
    use_time_bias = getattr(model, "use_time_bias", False)

    # Precompute RoPE tables if model uses RoPE
    rope_cos_eval = None
    rope_sin_eval = None
    if use_rope and hasattr(model, "rope_cos_table"):
        rope_cos_eval = model.rope_cos_table.unsqueeze(0)  # [1, L, D_head]
        rope_sin_eval = model.rope_sin_table.unsqueeze(0)

    @torch.no_grad()
    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)

        seq_input = valid[-MAX_LEN:]
        seq_len = len(seq_input)
        padded = [0] * (MAX_LEN - seq_len) + seq_input

        inter = Interaction({
            ITEM_SEQ_FIELD: torch.tensor([padded], dtype=torch.long, device=device),
            ITEM_LEN_FIELD: torch.tensor([seq_len], dtype=torch.long, device=device),
        })

        item_seq = inter[ITEM_SEQ_FIELD]  # [1, L]

        # Item embeddings + feature injection (same as SASRecF)
        item_emb = model.item_embedding(item_seq)
        sparse_emb, dense_emb = model.feature_embed_layer(None, item_seq)
        sparse_emb = sparse_emb["item"]
        dense_emb = dense_emb["item"]
        feature_table = []
        if sparse_emb is not None:
            feature_table.append(sparse_emb)
        if dense_emb is not None:
            feature_table.append(dense_emb)
        feature_table = torch.cat(feature_table, dim=-2)
        table_shape = feature_table.shape
        feat_num, embed_size = table_shape[-2], table_shape[-1]
        feature_emb = feature_table.view(
            table_shape[:-2] + (feat_num * embed_size,)
        )
        input_concat = torch.cat((item_emb, feature_emb), -1)
        concat_layer = getattr(model, "feat_concat_layer", None) or getattr(model, "concat_layer", None)
        input_emb = concat_layer(input_concat)

        # Position encoding
        extra_kwargs = {}
        if not use_rope:
            position_ids = torch.arange(
                item_seq.size(1), dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
            position_embedding = model.position_embedding(position_ids)
            input_emb = input_emb + position_embedding
        else:
            L = item_seq.size(1)
            extra_kwargs["rope_cos"] = rope_cos_eval[:, :L, :].to(device)
            extra_kwargs["rope_sin"] = rope_sin_eval[:, :L, :].to(device)

        # Time-interval bias
        if use_time_bias and history_ts is not None:
            ts_input = history_ts[-MAX_LEN:]  # align with padded sequence
            ts_padded = [0.0] * (MAX_LEN - len(ts_input)) + ts_input
            ts_tensor = torch.tensor([ts_padded], dtype=torch.float, device=device)
            time_bias = compute_time_bias(
                ts_tensor,
                method=getattr(model, "time_bias_method", "log"),
                weight=getattr(model, "time_bias_weight", torch.tensor(0.01)),
            )
            extra_kwargs["time_bias"] = time_bias

        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)

        extended_attention_mask = model.get_attention_mask(item_seq)
        trm_output = model.trm_encoder(
            input_emb, extended_attention_mask,
            output_all_encoded_layers=True, **extra_kwargs
        )
        output = trm_output[-1]

        mask = (item_seq != 0).float().unsqueeze(-1)
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        test_items_emb = model.item_embedding.weight[:n_items]
        scores = torch.matmul(pooled, test_items_emb.transpose(0, 1))
        return scores.squeeze(0).cpu().numpy()

    return score


def build_sasrec_scorer(model, n_items, config, device):
    """SASRec: unidirectional Transformer, mean-pool over non-padding positions.

    Replicates SASRec.forward() but skips gather_indexes(last_pos) —
    mean-pools all positions instead, same paradigm as BERT4Rec temporal eval.
    """
    MAX_LEN = config["MAX_ITEM_LIST_LENGTH"]
    ITEM_SEQ_FIELD = config["ITEM_ID_FIELD"] + config["LIST_SUFFIX"]
    ITEM_LEN_FIELD = config["ITEM_LIST_LENGTH_FIELD"]

    from recbole.data.interaction import Interaction

    @torch.no_grad()
    def score(history_ids, history_ts=None):
        if not history_ids:
            return np.zeros(n_items)
        valid = [i for i in history_ids if 0 < i < n_items]
        if not valid:
            return np.zeros(n_items)

        seq_input = valid[-MAX_LEN:]
        seq_len = len(seq_input)
        padded = [0] * (MAX_LEN - seq_len) + seq_input

        inter = Interaction({
            ITEM_SEQ_FIELD: torch.tensor([padded], dtype=torch.long, device=device),
            ITEM_LEN_FIELD: torch.tensor([seq_len], dtype=torch.long, device=device),
        })

        item_seq = inter[ITEM_SEQ_FIELD]  # [1, L]

        # SASRec embedding + position
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = model.position_embedding(position_ids)
        item_emb_seq = model.item_embedding(item_seq)
        input_emb = item_emb_seq + position_embedding
        input_emb = model.LayerNorm(input_emb)
        input_emb = model.dropout(input_emb)

        # Causal attention mask (unidirectional: left-to-right)
        extended_attention_mask = model.get_attention_mask(item_seq)

        # Forward through Transformer encoder
        trm_output = model.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]  # [1, L, H] — take last layer

        # Mean-pool over non-padding positions
        mask = (item_seq != 0).float().unsqueeze(-1)  # [1, L, 1]
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Dot-product with all item embeddings (no output_bias, unlike BERT4Rec)
        test_items_emb = model.item_embedding.weight[:n_items]
        scores = torch.matmul(pooled, test_items_emb.transpose(0, 1))
        return scores.squeeze(0).cpu().numpy()

    return score


# ── Evaluation ─────────────────────────────────────────────────────────────

def temporal_eval(user_histories, user_timestamps, user_evals, scoring_fn, n_items):
    """Evaluate: history → score → metrics on eval positives only."""
    rng = np.random.RandomState(SEED)
    all_user_scores = []
    stats = {"no_history": 0, "no_pos": 0, "no_neg": 0, "valid": 0}

    for uid, ev in tqdm(user_evals.items(), desc="Temporal eval", unit="user"):
        history = user_histories.get(uid, [])
        if not history:
            stats["no_history"] += 1
            continue

        positives = ev["positives"]
        hard_negs = ev["hard_negs"]
        if not positives:
            stats["no_pos"] += 1
            continue
        if len(hard_negs) < NUM_NEG:
            stats["no_neg"] += 1
            continue
        stats["valid"] += 1

        fixed_negs = sorted(rng.choice(hard_negs, NUM_NEG, replace=False))

        # Pass timestamps if available (for time-bias models)
        history_ts = user_timestamps.get(uid)
        all_scores = scoring_fn(history, history_ts)

        user_pos_scores = []
        for pos in positives:
            cand = [pos] + fixed_negs
            cand_scores = all_scores[cand]
            user_pos_scores.append(cand_scores)

        all_user_scores.append(np.array(user_pos_scores))

    if not all_user_scores:
        print(f"No valid users. Stats: {stats}")
        return OrderedDict([("error", "no_valid_users")]), stats

    return _compute_metrics(all_user_scores), stats


def _compute_metrics(all_user_scores):
    """GAUC, Recall@K, NDCG@K."""
    results = OrderedDict()
    all_aucs = []
    for user_scores in all_user_scores:
        user_auc = 0.0
        for i in range(user_scores.shape[0]):
            scores = user_scores[i]
            user_auc += np.mean(scores[1:] < scores[0])
        all_aucs.append(user_auc / user_scores.shape[0])
    results["GAUC"] = float(np.mean(all_aucs))

    for k in K_VALUES:
        recall_vals, ndcg_vals = [], []
        for user_scores in all_user_scores:
            user_recall, user_ndcg = 0.0, 0.0
            for i in range(user_scores.shape[0]):
                scores = user_scores[i]
                rank = np.sum(scores > scores[0])
                if rank < k:
                    user_recall += 1.0
                    user_ndcg += 1.0 / np.log2(rank + 2)
            recall_vals.append(user_recall / user_scores.shape[0])
            ndcg_vals.append(user_ndcg / user_scores.shape[0])
        results[f"Recall@{k}"] = float(np.mean(recall_vals))
        results[f"NDCG@{k}"] = float(np.mean(ndcg_vals))

    results["eval_users"] = len(all_user_scores)
    return results


# ── Main ────────────────────────────────────────────────────────────────────

SCORER_BUILDERS = {
    "BERT4Rec": "build_bert4rec_scorer",
    "BERT4RecF": "build_bert4rec_scorer",       # forward() handles features internally
    "BERT4RecP": "build_bert4rec_scorer",       # forward() handles innovations internally
    "NeuMF": "build_neumf_scorer",
    "BPR": "build_bpr_scorer",
    "ItemKNN": "build_itemknn_scorer",
    "DeepFM": "build_deepfm_scorer",
    "DeepFM_Feat": "build_deepfm_feat_scorer",
    "SASRec": "build_sasrec_scorer",
    "SASRecF": "build_sasrecf_scorer",
    "SASRecP": "build_sasrecp_scorer",
}


def main():
    parser = argparse.ArgumentParser(
        description="Temporal leave-one-out evaluation (no data leakage)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=list(SCORER_BUILDERS.keys()))
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_path = f"dataset/kuairec/{args.split}_data.pkl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print(f"Loading: {args.checkpoint}")
    config, model, dataset = load_model_and_data(args.checkpoint, args.device)
    n_items = dataset.item_num
    vid2int = dataset.field2token_id[config["ITEM_ID_FIELD"]]
    print(f"Model: {args.model} | Items: {n_items:,}")

    with open(data_path, "rb") as f:
        test_data = pickle.load(f)
    print(f"Data: {len(test_data):,} rows, {test_data['user_id'].nunique():,} users")

    # Build temporal splits (no leakage!)
    print("Building temporal splits (history 80% / eval 20%)...")
    user_histories, user_timestamps, user_evals = build_temporal_splits(test_data, vid2int)
    eval_users = len(user_evals)
    total_hist = sum(len(h) for h in user_histories.values())
    total_eval_pos = sum(len(ev["positives"]) for ev in user_evals.values())
    print(f"  Users with valid eval data: {eval_users}")
    print(f"  Total history items: {total_hist:,}")
    print(f"  Total eval positives: {total_eval_pos:,}")

    # Build scoring function
    print(f"\nBuilding scorer for {args.model}...")
    if args.model in ("BERT4Rec", "BERT4RecF", "BERT4RecP"):
        scoring_fn = build_bert4rec_scorer(model, n_items, config, args.device)
    elif args.model == "SASRec":
        scoring_fn = build_sasrec_scorer(model, n_items, config, args.device)
    elif args.model == "SASRecF":
        scoring_fn = build_sasrecf_scorer(model, n_items, config, args.device)
    elif args.model == "SASRecP":
        scoring_fn = build_sasrecp_scorer(model, n_items, config, args.device)
    elif args.model == "NeuMF":
        scoring_fn = build_neumf_scorer(model, n_items, args.device)
    elif args.model == "BPR":
        scoring_fn = build_bpr_scorer(model, n_items, args.device)
    elif args.model == "ItemKNN":
        scoring_fn = build_itemknn_scorer(model, n_items, args.device)
    elif args.model == "DeepFM":
        scoring_fn = build_deepfm_scorer(model, n_items, dataset, args.device)
    elif args.model == "DeepFM_Feat":
        scoring_fn = build_deepfm_feat_scorer(model, n_items, dataset, args.device)

    # Evaluate
    print(f"Running temporal evaluation ({args.split})...")
    results, stats = temporal_eval(user_histories, user_timestamps, user_evals, scoring_fn, n_items)

    # Print
    print("\n" + "=" * 60)
    print(f"{args.model} TEMPORAL Results ({args.split}, history 80% → eval 20%)")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")
    print(f"  {'user_stats':20s}: valid={stats.get('valid','?')}, "
          f"no_hist={stats.get('no_history','?')}, "
          f"no_pos={stats.get('no_pos','?')}, "
          f"no_neg={stats.get('no_neg','?')}")
    print("=" * 60)

    # Save
    os.makedirs("log", exist_ok=True)
    result_file = f"log/eval_TEMPORAL_{args.model}_{args.split}.txt"
    with open(result_file, "w") as f:
        f.write(f"{args.model} TEMPORAL ({args.split}, history 80% → eval 20%)\n")
        f.write("=" * 50 + "\n")
        f.write(f"Eval users: {eval_users}\n")
        f.write(f"History items: {total_hist:,}\n")
        f.write(f"Eval positives: {total_eval_pos:,}\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
        f.write(f"user_stats: {stats}\n")
    print(f"Saved to {result_file}")


if __name__ == "__main__":
    main()
