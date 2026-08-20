#!/usr/bin/env python3
"""
Utility script to analyze and visualize MCTS training data from saved checkpoints and CSV files.
"""

import os
import csv
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def load_csv_metrics(csv_path):
    """Load metrics from CSV file."""
    metrics = {
        'iteration': [],
        'avg_loss': [],
        'value_loss': [],
        'policy_loss': [],
        'game_length': [],
        'win_rate': [],
        'learning_rate': [],
    }
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return metrics
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in metrics:
                val = row.get(key)
                if val and val != 'None':
                    try:
                        metrics[key].append(float(val))
                    except ValueError:
                        pass
    
    return metrics


def load_checkpoint_info(checkpoint_path):
    """Load information from a checkpoint file."""
    try:
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        return {
            'iteration': data['iteration'],
            'timestamp': data['timestamp'],
            'W_shapes': [w.shape for w in data['weights_W']],
            'W_out_shape': data['weights_W_out'].shape,
        }
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def list_checkpoints(data_dir="MCTS_data"):
    """List all available checkpoints."""
    checkpoint_dir = Path(data_dir)
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_iter_*.pkl"))
    
    print(f"\nAvailable checkpoints in {data_dir}:")
    print(f"{'Iteration':<12} {'File':<30}")
    print("-" * 45)
    
    for cp in checkpoints:
        info = load_checkpoint_info(cp)
        if info:
            print(f"{info['iteration']:<12} {cp.name:<30}")
    
    if not checkpoints:
        print("No checkpoints found")
    
    return checkpoints


def plot_metrics(csv_path=None, data_dir="MCTS_data"):
    """Generate comprehensive training visualization."""
    if csv_path is None:
        csv_path = os.path.join(data_dir, "training_metrics.csv")
    
    metrics = load_csv_metrics(csv_path)
    
    if not metrics['iteration']:
        print("No metrics data found")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"MCTS Training Analysis - {data_dir}")
    
    # Loss plot
    axes[0, 0].plot(metrics['iteration'], metrics['avg_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Average Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Value vs Policy loss
    axes[0, 1].plot(metrics['iteration'], metrics['value_loss'], 'g-', label='Value Loss', linewidth=2)
    axes[0, 1].plot(metrics['iteration'], metrics['policy_loss'], 'r-', label='Policy Loss', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 2].plot(metrics['iteration'], metrics['learning_rate'], 'k-', linewidth=2)
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Game length
    if metrics['game_length']:
        axes[1, 0].plot(range(len(metrics['game_length'])), metrics['game_length'], 'm-', linewidth=2)
        axes[1, 0].set_xlabel('Game Index')
        axes[1, 0].set_ylabel('Moves')
        axes[1, 0].set_title('Game Length Over Time')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Win rate
    if metrics['win_rate']:
        axes[1, 1].plot(range(len(metrics['win_rate'])), metrics['win_rate'], 'c-', linewidth=2)
        axes[1, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50% (Neutral)')
        axes[1, 1].set_xlabel('Game Index')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].set_title('Win Rate Over Time')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    game_stats = ""
    if metrics['game_length']:
        game_stats += f"\nAvg Game Length: {np.mean(metrics['game_length']):.1f}\nMax Game Length: {max(metrics['game_length']):.0f}"
    
    win_stats = ""
    if metrics['win_rate']:
        win_stats += f"\nAvg Win Rate: {np.mean(metrics['win_rate']):.2f}"
    
    summary_text = f"""Training Summary

Iterations: {len(metrics['iteration'])}
Final Loss: {metrics['avg_loss'][-1]:.4f}
Min Loss: {min(metrics['avg_loss']):.4f}
Max Loss: {max(metrics['avg_loss']):.4f}

Final LR: {metrics['learning_rate'][-1]:.6f}
Initial LR: {metrics['learning_rate'][0]:.6f}

Games Tracked: {len(metrics['game_length'])}{game_stats}{win_stats}
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def generate_report(data_dir="MCTS_data", report_path=None):
    """Generate a training report."""
    csv_path = os.path.join(data_dir, "training_metrics.csv")
    metrics = load_csv_metrics(csv_path)
    
    report = {
        'timestamp': str(Path(data_dir).stat().st_mtime),
        'total_iterations': len(metrics['iteration']),
        'final_loss': float(metrics['avg_loss'][-1]) if metrics['avg_loss'] else None,
        'min_loss': float(min(metrics['avg_loss'])) if metrics['avg_loss'] else None,
        'max_loss': float(max(metrics['avg_loss'])) if metrics['avg_loss'] else None,
        'avg_loss': float(np.mean(metrics['avg_loss'])) if metrics['avg_loss'] else None,
        'final_lr': float(metrics['learning_rate'][-1]) if metrics['learning_rate'] else None,
        'total_games': len(metrics['game_length']),
        'avg_game_length': float(np.mean(metrics['game_length'])) if metrics['game_length'] else None,
        'avg_win_rate': float(np.mean(metrics['win_rate'])) if metrics['win_rate'] else None,
    }
    
    if report_path is None:
        report_path = RESULTS_DIR / "training_report.json"
    else:
        report_path = Path(report_path)
        if not report_path.is_absolute():
            report_path = RESULTS_DIR / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report


def print_report(data_dir="MCTS_data", report_path=None):
    """Print a formatted training report."""
    report = generate_report(data_dir, report_path)
    
    print(f"\n{'='*50}")
    print(f"Training Report - {data_dir}")
    print(f"{'='*50}")
    print(f"Total Iterations: {report['total_iterations']}")
    print(f"Total Games: {report['total_games']}")
    print(f"\nLoss Metrics:")
    final_loss_str = f"{report['final_loss']:.6f}" if report['final_loss'] else "N/A"
    min_loss_str = f"{report['min_loss']:.6f}" if report['min_loss'] else "N/A"
    max_loss_str = f"{report['max_loss']:.6f}" if report['max_loss'] else "N/A"
    avg_loss_str = f"{report['avg_loss']:.6f}" if report['avg_loss'] else "N/A"
    print(f"  Final:   {final_loss_str}")
    print(f"  Min:     {min_loss_str}")
    print(f"  Max:     {max_loss_str}")
    print(f"  Average: {avg_loss_str}")
    print(f"\nTraining Parameters:")
    final_lr_str = f"{report['final_lr']:.8f}" if report['final_lr'] else "N/A"
    print(f"  Final LR: {final_lr_str}")
    print(f"\nGame Statistics:")
    if report['avg_game_length']:
        print(f"  Avg Length: {report['avg_game_length']:.1f} moves")
    else:
        print(f"  Avg Length: N/A")
    
    if report['avg_win_rate']:
        print(f"  Avg Win Rate: {report['avg_win_rate']:.2%}")
    else:
        print(f"  Avg Win Rate: N/A")
    print(f"{'='*50}\n")
    
    return report


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MCTS training data")
    parser.add_argument("--data-dir", type=str, default="MCTS_data", help="Data directory")
    parser.add_argument("--report-path", type=str, default=str(RESULTS_DIR / "training_report.json"), help="Output report file path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--report", action="store_true", help="Generate report")
    parser.add_argument("--list-checkpoints", action="store_true", help="List checkpoints")
    parser.add_argument("--all", action="store_true", help="Do everything")
    
    args = parser.parse_args()
    
    if args.all or args.list_checkpoints:
        list_checkpoints(args.data_dir)
    
    if args.all or args.report:
        print_report(args.data_dir, args.report_path)
    
    if args.all or args.plot:
        fig = plot_metrics(data_dir=args.data_dir)
        if fig:
            plt.show()


if __name__ == "__main__":
    main()
