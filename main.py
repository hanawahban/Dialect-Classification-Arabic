import os
import sys

from src.preprocess import run as preprocess
from src.train_classical import run as train_classical
from src.evaluate import run as evaluate
from config import FIGURES_DIR, METRICS_DIR


def ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)


if __name__ == '__main__':
    print("=" * 60)
    print("ARABIC DIALECT IDENTIFICATION — CLASSICAL PIPELINE")
    print("=" * 60)

    ensure_dirs()

    print("\n[1/3] Preprocessing...")
    train, test = preprocess()

    print("\n[2/3] Training classical models...")
    results = train_classical(train, test)

    print("\n[3/3] Generating evaluation figures and metrics...")
    summary = evaluate(results)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(summary.to_string(index=False))
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Metrics saved to: {METRICS_DIR}")