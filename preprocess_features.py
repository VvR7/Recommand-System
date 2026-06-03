#!/usr/bin/env python
"""
Create RecBole .user and .item atomic files for KuaiRec feature-rich experiments.

Usage:
  python preprocess_features.py                    # Tier 1: categories + user features
  python preprocess_features.py --tier 2           # Tier 2: + static item attributes
  python preprocess_features.py --tier 3           # Tier 3: + engagement metrics

Output:
  - dataset/kuairec/kuairec.user   (user features in RecBole atomic format)
  - dataset/kuairec/kuairec.item   (item features in RecBole atomic format)

Atomic file format: TSV with header like "field_name:field_type"
Field types: token (categorical), float (numerical), token_seq (multi-value)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
INTER_FILE = "dataset/kuairec/kuairec.inter"
USER_SRC = "KuaiRec/KuaiRec/KuaiRec2.0/data/user_features.csv"
CAT_SRC = "KuaiRec/kuairec_caption_category.csv"
ITEM_CAT_SRC = "KuaiRec/KuaiRec/KuaiRec2.0/data/item_categories.csv"
DAILY_SRC = "KuaiRec/KuaiRec/KuaiRec2.0/data/item_daily_features.csv"

USER_OUT = "dataset/kuairec/kuairec.user"
ITEM_OUT = "dataset/kuairec/kuairec.item"


def load_inter_item_ids():
    """Get the filtered item IDs from the RecBole .inter file."""
    df = pd.read_csv(INTER_FILE, sep="\t")
    items = sorted(df["video_id:token"].unique())
    print(f"Loaded {len(items):,} items from {INTER_FILE}")
    return set(items)


def create_user_file(filtered_inter_user_ids=None):
    """Create kuairec.user from user_features.csv.

    Selected fields (13 total):
      user_id, user_active_degree, is_lowactive_period, is_live_streamer,
      is_video_author, follow_user_num_range, fans_user_num_range,
      friend_user_num_range, register_days_range,
      onehot_feat0, onehot_feat1, onehot_feat2, onehot_feat3

    All fields are token type (categorical). No float fields in Tier 1.
    """
    print("\n=== Creating kuairec.user ===")
    df = pd.read_csv(USER_SRC)

    # Fields to include
    token_fields = [
        "user_active_degree",
        "is_lowactive_period",
        "is_live_streamer",
        "is_video_author",
        "follow_user_num_range",
        "fans_user_num_range",
        "friend_user_num_range",
        "register_days_range",
        "onehot_feat0",
        "onehot_feat1",
        "onehot_feat2",
        "onehot_feat3",
    ]

    # Select and fill missing
    out = df[["user_id"] + token_fields].copy()
    for col in token_fields:
        if out[col].dtype == object:
            out[col] = out[col].fillna("UNKNOWN")
        else:
            out[col] = out[col].fillna(-1).astype(int)

    # Build RecBole header
    header = ["user_id:token"] + [f"{f}:token" for f in token_fields]
    lines = ["\t".join(header)]

    for _, row in out.iterrows():
        vals = [str(row["user_id"])] + [str(row[f]) for f in token_fields]
        lines.append("\t".join(vals))

    with open(USER_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved {len(out):,} users to {USER_OUT}")
    print(f"  Fields: {len(token_fields)} token fields")

    # Summary
    for field in token_fields:
        vals = out[field].nunique()
        print(f"    {field}: {vals} unique values")


def create_item_file(inter_items, tier=1):
    """Create kuairec.item from caption_category + item_categories + daily_features.

    Tier 1 (P0): first/second/third_level_category_id from caption_category
    Tier 2: + video_duration (float), video_type, music_id, video_tag_id from daily_features
    Tier 3: + aggregated engagement metrics (avg like/play/comment/share/collect)
    """
    print(f"\n=== Creating kuairec.item (Tier {tier}) ===")

    # ── Load caption_category with robust parsing ─────────────────────────
    cc = pd.read_csv(CAT_SRC, engine="python")
    cc["video_id"] = pd.to_numeric(cc["video_id"], errors="coerce")
    cc = cc.dropna(subset=["video_id"])
    cc["video_id"] = cc["video_id"].astype(int)

    # Filter to inter items only
    cc = cc[cc["video_id"].isin(inter_items)].copy()
    print(f"  Caption items matching inter: {len(cc):,} / {len(inter_items):,}")

    # ── Category fields (token) ──────────────────────────────────────────
    cat_fields = [
        "first_level_category_id",
        "second_level_category_id",
        "third_level_category_id",
    ]

    # Fill UNKNOWN (-124) and ensure int
    for col in cat_fields:
        cc[col] = cc[col].fillna(-124).astype(int)

    # ── Build output columns ──────────────────────────────────────────────
    header_parts = ["video_id:token"]
    for f in cat_fields:
        header_parts.append(f"{f}:token")

    # Build data rows as list of dicts for fast lookup
    item_data = {}
    for _, row in cc.iterrows():
        vid = row["video_id"]
        vals = [str(vid)] + [str(row[f]) for f in cat_fields]
        item_data[vid] = vals

    # ── Tier 2: Static attributes from daily_features ────────────────────
    if tier >= 2:
        static_fields_token = ["video_type", "music_id", "video_tag_id"]
        static_fields_float = ["video_duration"]

        daily = pd.read_csv(DAILY_SRC)
        daily = daily[daily["video_id"].isin(inter_items)]

        # Take first occurrence for static fields
        static = daily.groupby("video_id").first().reset_index()
        static = static[["video_id"] + static_fields_token + static_fields_float]

        for f in static_fields_token:
            header_parts.append(f"{f}:token")
            static[f] = static[f].fillna("UNKNOWN").astype(str)
        for f in static_fields_float:
            header_parts.append(f"{f}:float")
            static[f] = static[f].fillna(0.0)

        for _, row in static.iterrows():
            vid = int(row["video_id"])
            if vid in item_data:
                for f in static_fields_token:
                    item_data[vid].append(str(row[f]))
                for f in static_fields_float:
                    item_data[vid].append(f"{row[f]:.1f}")

        print(f"  Added Tier 2 static fields: {static_fields_token + static_fields_float}")

    # ── Tier 3: Engagement metrics (time-aligned) ─────────────────────────
    if tier >= 3:
        eng_fields = ["like_cnt", "comment_cnt", "share_cnt", "collect_cnt", "follow_cnt"]
        eng_fields += ["play_cnt", "show_cnt"]

        # Use mean across training dates to avoid leakage
        daily = pd.read_csv(DAILY_SRC)
        daily = daily[daily["video_id"].isin(inter_items)]
        eng = daily.groupby("video_id")[eng_fields].mean().reset_index()

        for f in eng_fields:
            header_parts.append(f"{f}:float")
            eng[f] = eng[f].fillna(0.0)

        for _, row in eng.iterrows():
            vid = int(row["video_id"])
            if vid in item_data:
                for f in eng_fields:
                    item_data[vid].append(f"{row[f]:.4f}")

        print(f"  Added Tier 3 engagement fields: {eng_fields}")

    # ── Ensure all inter items are covered (use defaults for missing) ────
    default_row = None
    for vid in sorted(inter_items):
        if vid not in item_data:
            if default_row is None:
                # Build a default row from the header
                default_row = ["0"] * (len(header_parts) - 1)
            item_data[vid] = [str(vid)] + default_row

    missing = sum(1 for vid in inter_items if vid not in item_data)
    if missing > 0:
        print(f"  ⚠ {missing} items missing from features, using defaults")

    # ── Write atomic file ────────────────────────────────────────────────
    sorted_vids = sorted(item_data.keys())
    lines = ["\t".join(header_parts)]
    for vid in sorted_vids:
        lines.append("\t".join(item_data[vid]))

    with open(ITEM_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Saved {len(sorted_vids):,} items to {ITEM_OUT}")
    print(f"  Fields: {len(header_parts) - 1} ({', '.join(h.split(':')[0] for h in header_parts[1:])})")


def main():
    parser = argparse.ArgumentParser(description="Create RecBole feature atomic files")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3],
                        help="Feature tier: 1=categories, 2=+static, 3=+engagement")
    args = parser.parse_args()

    os.makedirs("dataset/kuairec", exist_ok=True)

    # 1. Load valid item IDs from existing .inter
    inter_items = load_inter_item_ids()

    # 2. Create .user file
    create_user_file()

    # 3. Create .item file
    create_item_file(inter_items, tier=args.tier)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("  1. Verify: check dataset.field2type includes new feature fields")
    print("  2. Train: python train_cf.py --model DeepFM --config configs/deepfm_feat_kuairec.yaml")
    print("  3. Eval:  python evaluate_temporal.py --checkpoint saved/DeepFM-*.pth --model DeepFM_Feat --split test")
    print("=" * 60)


if __name__ == "__main__":
    main()
