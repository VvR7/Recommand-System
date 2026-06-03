#!/usr/bin/env python
"""
Train CF/MF baseline models (ItemKNN, BPR) on KuaiRec using RecBole.

Usage:
  python train_cf.py --model ItemKNN                   # Item-based CF
  python train_cf.py --model BPR                       # BPR-MF
  python train_cf.py --model ItemKNN --smoke           # Quick smoke test
  python train_cf.py --model BPR --epochs 100 --lr 0.0005

Output:
  - saved/<Model>-<timestamp>.pth  (best checkpoint)
  - log/train_cf.log               (training log)

VRAM requirements (local 8GB GPU):
  - ItemKNN: <1 GB (similarity computed on CPU)
  - BPR:     <2 GB (small embedding tables)
"""

import argparse
import os
import sys
import time
import logging
from datetime import datetime

os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

# ── Logging ────────────────────────────────────────────────────────────────
os.makedirs("log", exist_ok=True)
LOG_FILE = "log/train_cf.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_cf")

# ── Model Registry ─────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "ItemKNN": "configs/itemknn_kuairec.yaml",
    "BPR": "configs/bpr_kuairec.yaml",
    "NeuMF": "configs/neumf_kuairec.yaml",
    "DeepFM": "configs/deepfm_kuairec.yaml",
    "SASRec": "configs/sasrec_kuairec.yaml",
    "SASRecF": "configs/sasrecf_feat_kuairec.yaml",
    "SASRecP": "configs/sasrecp_kuairec.yaml",
    "BERT4RecF": "configs/bert4recf_feat_kuairec.yaml",
    "BERT4RecP": "configs/bert4recp_kuairec.yaml",
}


def main():
    parser = argparse.ArgumentParser(
        description="Train CF/MF baseline on KuaiRec")
    parser.add_argument("--model", type=str, required=True,
                        help="Model to train (RecBole class name)")
    parser.add_argument("--config", type=str, default=None,
                        help="Override config file path (for feature variants)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: few epochs, small batch")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config_file = args.config if args.config else MODEL_CONFIGS[args.model]
    if not os.path.exists(config_file):
        logger.error(f"Config not found: {config_file}")
        sys.exit(1)

    # Verify data
    inter_file = "dataset/kuairec/kuairec.inter"
    if not os.path.exists(inter_file):
        logger.error(f"Data not found: {inter_file}. Run preprocess.py first.")
        sys.exit(1)

    # ── Header ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"{args.model} Training on KuaiRec")
    logger.info(f"Start:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Config: {config_file}")
    logger.info(f"GPU:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info("=" * 60)

    # ── Build config overrides ──────────────────────────────────────────
    # NOTE: eval mode is left to each config YAML (full for BPR/ItemKNN, labeled for NeuMF)
    config_overrides = {}

    if args.smoke:
        logger.info("*** SMOKE TEST MODE ***")
        config_overrides.update({
            "epochs": 3,
            "stopping_step": 3,
            "train_batch_size": 256,
            "eval_batch_size": 512,
            "show_progress": True,
        })
    else:
        config_overrides["show_progress"] = True
        if args.epochs:
            config_overrides["epochs"] = args.epochs
        if args.lr:
            config_overrides["learning_rate"] = args.lr
        if args.batch_size:
            config_overrides["train_batch_size"] = args.batch_size

    # ── Print summary ───────────────────────────────────────────────────
    logger.info("Configuration:")
    logger.info(f"  Model:         {args.model}")
    logger.info(f"  Epochs:        {config_overrides.get('epochs', 'from config')}")
    logger.info(f"  Learning rate: {config_overrides.get('learning_rate', 'from config')}")
    logger.info(f"  Batch size:    {config_overrides.get('train_batch_size', 'from config')}")

    # ── Train ───────────────────────────────────────────────────────────
    from recbole.quick_start import run_recbole

    t_start = time.time()
    try:
        result = run_recbole(
            model=args.model,
            dataset="kuairec",
            config_file_list=[config_file],
            config_dict=config_overrides,
        )
    except torch.cuda.OutOfMemoryError:
        logger.error("GPU OUT OF MEMORY! Reduce --batch_size.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    total_time = time.time() - t_start

    # ── Results ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info(f"{args.model} TRAINING COMPLETE!")
    logger.info(f"Total wall time:    {total_time/3600:.2f}h ({total_time/60:.1f}min)")
    logger.info(f"Best valid score:   {result.get('best_valid_score', 'N/A')}")
    best_result = result.get('best_valid_result') or {}
    logger.info(f"Best valid result:  {dict(best_result)}")
    logger.info(f"Test result (big):  {dict(result.get('test_result', {}))}")
    logger.info(f"Checkpoint dir:     saved/")
    logger.info(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Save summary
    summary_file = f"log/train_{args.model}_results.txt"
    with open(summary_file, "w") as f:
        f.write(f"{args.model} on KuaiRec — Training Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Completed:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time:  {total_time/3600:.2f}h ({total_time/60:.1f}min)\n")
        f.write(f"Best valid:  {result.get('best_valid_score', 'N/A')}\n")
        best_res = result.get('best_valid_result') or {}
        test_res = result.get('test_result') or {}
        f.write(f"Best result: {dict(best_res)}\n")
        f.write(f"Test result: {dict(test_res)}\n")
    logger.info(f"Summary saved to {summary_file}")

    return result


if __name__ == "__main__":
    import torch
    main()
