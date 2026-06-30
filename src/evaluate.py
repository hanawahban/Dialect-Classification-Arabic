import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, METRICS_DIR, REGIONS_TO_KEEP


#Confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    ax.set_title(f'Confusion Matrix: {model_name}', fontsize=13)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    plt.tight_layout()

    filename = model_name.lower().replace(' ', '_') + '_confusion_matrix.png'
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


#Per-class metrics table
def save_metrics_table(y_true, y_pred, model_name, labels):
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )
    rows = []
    for label in labels:
        rows.append({
            'Dialect':   label,
            'Precision': round(report[label]['precision'], 4),
            'Recall':    round(report[label]['recall'],    4),
            'F1':        round(report[label]['f1-score'],  4),
            'Support':   int(report[label]['support'])
        })

    rows.append({
        'Dialect':   'Macro Avg',
        'Precision': round(report['macro avg']['precision'], 4),
        'Recall':    round(report['macro avg']['recall'],    4),
        'F1':        round(report['macro avg']['f1-score'],  4),
        'Support':   int(report['macro avg']['support'])
    })

    df = pd.DataFrame(rows)
    filename = model_name.lower().replace(' ', '_') + '_metrics.csv'
    path = os.path.join(METRICS_DIR, filename)
    df.to_csv(path, index=False)
    print(f"  Saved metrics table  → {path}")
    return df


#Accuracy comparison bar chart
def plot_accuracy_comparison(results):
    df = pd.DataFrame(results)
    n  = len(df)
    colors = ['#D65F5F' if i == n - 1 else '#4878CF' for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(n),
        df['accuracy'],
        color=colors,
        edgecolor='white',
        width=0.5
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy Comparison Across All Models', fontsize=13)
    ax.set_xticks(range(n))
    ax.set_xticklabels(df['model'], rotation=15, ha='right', fontsize=9)

    for bar, val in zip(bars, df['accuracy']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved accuracy chart → {path}")


#Macro F1 comparison bar chart
def plot_f1_comparison(results):
    df = pd.DataFrame(results)
    n  = len(df)
    colors = ['#D65F5F' if i == n - 1 else '#4878CF' for i in range(n)]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        range(n),
        df['f1'],
        color=colors,
        edgecolor='white',
        width=0.5
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Macro F1 Score', fontsize=11)
    ax.set_title('Macro F1 Comparison Across All Models', fontsize=13)
    ax.set_xticks(range(n))
    ax.set_xticklabels(df['model'], rotation=15, ha='right', fontsize=9)

    for bar, val in zip(bars, df['f1']):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'f1_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved F1 chart       → {path}")


#Training and inference time
def plot_time_comparison(results):
    df = pd.DataFrame(results)
    n  = len(df)
    colors = ['#D65F5F' if i == n - 1 else '#4878CF' for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Training time
    axes[0].bar(
        range(n), df['train_time'],
        color=colors, edgecolor='white', width=0.5
    )
    axes[0].set_ylabel('Time (seconds)', fontsize=11)
    axes[0].set_title('Training Time', fontsize=13)
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels(df['model'], rotation=15, ha='right', fontsize=9)

    # Inference time
    axes[1].bar(
        range(n), df['inference_time'],
        color=colors, edgecolor='white', width=0.5
    )
    axes[1].set_ylabel('Time (seconds)', fontsize=11)
    axes[1].set_title('Inference Time (Test Set)', fontsize=13)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels(df['model'], rotation=15, ha='right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'time_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved time chart     → {path}")


def run(results):
    """
    results: list of dicts, one per model, with keys:
        model, y_true, y_pred, accuracy, f1, train_time, inference_time
    """
    print("Generating evaluation figures and metrics...")
    labels = REGIONS_TO_KEEP

    summary = []
    for r in results:
        print(f"\n  [{r['model']}]")
        plot_confusion_matrix(r['y_true'], r['y_pred'], r['model'], labels)
        save_metrics_table(r['y_true'], r['y_pred'], r['model'], labels)
        summary.append({
            'model':          r['model'],
            'accuracy':       r['accuracy'],
            'f1':             r['f1'],
            'train_time':     r['train_time'],
            'inference_time': r['inference_time']
        })

    print("\n  [Overall comparison charts]")
    plot_accuracy_comparison(summary)
    plot_f1_comparison(summary)
    plot_time_comparison(summary)

    summary_df = pd.DataFrame(summary)
    path = os.path.join(METRICS_DIR, 'summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"  Saved summary        → {path}")

    print("\nEvaluation complete.")
    return summary_df


if __name__ == '__main__':
    print("Run main.py to execute the full pipeline.")