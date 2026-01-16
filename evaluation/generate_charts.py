"""
Generate charts from evaluation results.

Charts:
1. Accuracy comparison (bar chart)
2. Latency comparison (stacked bar)
3. Accuracy vs Latency tradeoff (scatter)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Configuration
COLORS = {
    'baseline': '#2196F3',   # Blue
    'flashrank': '#4CAF50',  # Green
    'pagerank': '#FF9800',   # Orange
}
LABELS = {
    'baseline': 'Baseline\n(ANN)',
    'flashrank': 'FlashRank\n(ANN+Rerank)',
    'pagerank': 'PageRank\n(ANN+Graph)',
}


def load_report(path: Path) -> Dict[str, Any]:
    """Load evaluation report."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_accuracy_chart(report: Dict[str, Any], output_path: Path):
    """Bar chart: Top-1 and Best-of-3 accuracy comparison."""
    metrics = report['metrics']
    methods = list(metrics.keys())

    top1_means = [metrics[m]['avg_top1_similarity'] for m in methods]
    top1_stds = [metrics[m]['std_top1_similarity'] for m in methods]
    best3_means = [metrics[m]['avg_best3_similarity'] for m in methods]
    best3_stds = [metrics[m]['std_best3_similarity'] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, top1_means, width, yerr=top1_stds,
                   label='Top-1', color=[COLORS.get(m, '#888') for m in methods],
                   capsize=4, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, best3_means, width, yerr=best3_stds,
                   label='Best-of-3', color=[COLORS.get(m, '#888') for m in methods],
                   capsize=4, edgecolor='black', linewidth=1, alpha=0.6, hatch='//')

    ax.set_ylabel('Cosine Similarity', fontweight='bold')
    ax.set_title('Retrieval Accuracy: Top-1 vs Best-of-3', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods])
    ax.set_ylim(0, 1.0)
    ax.legend()

    for bar, val in zip(bars1, top1_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, best3_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def create_latency_chart(report: Dict[str, Any], output_path: Path):
    """Stacked bar chart: Retrieval vs reranking latency."""
    metrics = report['metrics']
    methods = list(metrics.keys())

    retrieval = [metrics[m]['avg_retrieval_latency_ms'] for m in methods]
    rerank = [metrics[m]['avg_rerank_latency_ms'] for m in methods]
    total = [metrics[m]['avg_total_latency_ms'] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.6

    ax.bar(x, retrieval, width, label='Retrieval (ANN)', color='#64B5F6')
    ax.bar(x, rerank, width, bottom=retrieval, label='Reranking', color='#FF8A65')

    ax.set_ylabel('Latency (ms)', fontweight='bold')
    ax.set_title('Pipeline Latency Breakdown', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in methods])
    ax.legend(loc='upper left')

    for i, t in enumerate(total):
        ax.text(x[i], t + 5, f'{t:.0f}ms', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def create_tradeoff_chart(report: Dict[str, Any], output_path: Path):
    """Scatter plot: Accuracy vs latency tradeoff."""
    metrics = report['metrics']
    methods = list(metrics.keys())

    accuracy = [metrics[m]['avg_top1_similarity'] for m in methods]
    latency = [metrics[m]['avg_total_latency_ms'] for m in methods]

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, m in enumerate(methods):
        ax.scatter(latency[i], accuracy[i], s=200,
                   c=COLORS.get(m, '#888'),
                   edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(m.capitalize(), (latency[i], accuracy[i]),
                    xytext=(8, 4), textcoords='offset points', fontweight='bold')

    ax.set_xlabel('Total Latency (ms)', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy', fontweight='bold')
    ax.set_title('Accuracy vs Latency Tradeoff', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_table(report: Dict[str, Any]) -> str:
    """Create markdown table."""
    metrics = report['metrics']
    lines = [
        "| Method | Top-1 Sim | Best-3 Sim | Latency (ms) | Rerank (ms) |",
        "|--------|-----------|------------|--------------|-------------|"
    ]
    for m, data in metrics.items():
        lines.append(
            f"| {m.capitalize()} | "
            f"{data['avg_top1_similarity']:.4f} | "
            f"{data['avg_best3_similarity']:.4f} | "
            f"{data['avg_total_latency_ms']:.1f} | "
            f"{data['avg_rerank_latency_ms']:.1f} |"
        )
    return "\n".join(lines)


def generate_all(report_path: Optional[Path] = None, output_dir: Optional[Path] = None):
    """Generate all charts."""
    results_dir = Path(__file__).parent / "results"

    if report_path is None:
        reports = list(results_dir.glob("evaluation_*.json"))
        if not reports:
            print("No evaluation reports found. Run: python evaluation/run_evaluation.py")
            return
        report_path = max(reports, key=lambda p: p.stat().st_mtime)
        print(f"Using: {report_path}")

    if output_dir is None:
        output_dir = results_dir

    output_dir.mkdir(exist_ok=True)
    report = load_report(report_path)

    create_accuracy_chart(report, output_dir / "accuracy_comparison.png")
    create_latency_chart(report, output_dir / "latency_comparison.png")
    create_tradeoff_chart(report, output_dir / "tradeoff_analysis.png")

    print(f"\nAll charts saved to: {output_dir}")
    print("\nSummary:")
    print(create_summary_table(report))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation charts")
    parser.add_argument("--report", type=Path, help="Path to evaluation JSON")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()

    generate_all(args.report, args.output)
