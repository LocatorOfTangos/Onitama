#!/usr/bin/env python3
"""
Enhanced MCTS training script with data collection, visualization, and checkpointing.

Features:
- Collects loss, accuracy, and performance metrics
- Live visualization with matplotlib
- Periodic weight checkpointing
- Training progress logging to CSV
- Graceful handling of interruption with recovery
"""

import os
import csv
import json
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from collections import deque

from LeanBoardstate import Boardstate
from MCTS import NeuralNet, SelfPlayTrainer

# Detect if running in headless environment and set backend accordingly
try:
    import sys
    if 'DISPLAY' not in os.environ and sys.platform != 'darwin':
        matplotlib.use('Agg')  # Use non-interactive backend
except:
    pass


class TrainingMetrics:
    """Collects and manages training metrics."""
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.iteration = []
        self.avg_loss = []
        self.value_loss = []
        self.policy_loss = []
        self.game_length = []
        self.win_rate = []
        self.learning_rate = []
        
        # Windows for moving averages
        self.loss_window = deque(maxlen=window_size)
        self.value_window = deque(maxlen=window_size)
        self.policy_window = deque(maxlen=window_size)
        self.game_length_window = deque(maxlen=window_size)
        self.wins_window = deque(maxlen=window_size)
    
    def record_batch(self, iteration, batch_loss, value_loss, policy_loss, lr):
        """Record metrics for a training batch."""
        self.iteration.append(iteration)
        self.loss_window.append(batch_loss)
        self.value_window.append(value_loss)
        self.policy_window.append(policy_loss)
        self.learning_rate.append(lr)
        
        self.avg_loss.append(np.mean(self.loss_window))
        self.value_loss.append(np.mean(self.value_window))
        self.policy_loss.append(np.mean(self.policy_window))
    
    def record_game(self, game_length, winner_color, current_color):
        """Record metrics for a completed game."""
        self.game_length_window.append(game_length)
        is_win = 1.0 if winner_color == current_color else 0.0
        self.wins_window.append(is_win)
        
        self.game_length.append(np.mean(self.game_length_window))
        self.win_rate.append(np.mean(self.wins_window))
    
    def to_dict(self):
        """Convert metrics to dictionary for saving."""
        return {
            'iteration': self.iteration,
            'avg_loss': self.avg_loss,
            'value_loss': self.value_loss,
            'policy_loss': self.policy_loss,
            'game_length': self.game_length,
            'win_rate': self.win_rate,
            'learning_rate': self.learning_rate,
        }
    
    def from_dict(self, data):
        """Load metrics from dictionary."""
        self.iteration = data.get('iteration', [])
        self.avg_loss = data.get('avg_loss', [])
        self.value_loss = data.get('value_loss', [])
        self.policy_loss = data.get('policy_loss', [])
        self.game_length = data.get('game_length', [])
        self.win_rate = data.get('win_rate', [])
        self.learning_rate = data.get('learning_rate', [])


class EnhancedSelfPlayTrainer(SelfPlayTrainer):
    """Extended self-play trainer with data collection and visualization."""
    
    def __init__(self, net, mcts_simulations=100, replay_size=10000, data_dir="MCTS_data"):
        super().__init__(net, mcts_simulations=mcts_simulations, replay_size=replay_size)
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.metrics = TrainingMetrics()
        self.checkpoint_interval = 10  # Save weights every N iterations
        self.visualize_interval = 5    # Update plots every N iterations
        
        # Setup matplotlib for live plotting
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle("MCTS Training Progress")
    
    def save_checkpoint(self, iteration):
        """Save network weights to file."""
        checkpoint_path = os.path.join(self.data_dir, f"checkpoint_iter_{iteration:06d}.pkl")
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'weights_W': [w.copy() for w in self.net.W],
            'weights_b': [b.copy() for b in self.net.b],
            'weights_W_out': self.net.W_out.copy(),
            'weights_b_out': self.net.b_out.copy(),
            'metrics': self.metrics.to_dict(),
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def save_latest_checkpoint(self, iteration):
        """Save weights to 'latest' file for quick recovery."""
        latest_path = os.path.join(self.data_dir, "latest_checkpoint.pkl")
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'weights_W': [w.copy() for w in self.net.W],
            'weights_b': [b.copy() for b in self.net.b],
            'weights_W_out': self.net.W_out.copy(),
            'weights_b_out': self.net.b_out.copy(),
            'replay_buffer': list(self.replay.buffer),
            'metrics': self.metrics.to_dict(),
        }
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, checkpoint_path="latest"):
        """Load network weights from checkpoint."""
        if checkpoint_path == "latest":
            checkpoint_path = os.path.join(self.data_dir, "latest_checkpoint.pkl")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.net.W = checkpoint_data['weights_W']
        self.net.b = checkpoint_data['weights_b']
        self.net.W_out = checkpoint_data['weights_W_out']
        self.net.b_out = checkpoint_data['weights_b_out']
        
        self.replay.buffer.clear()
        for ex in checkpoint_data.get('replay_buffer', []):
            self.replay.push(ex)
        
        self.metrics.from_dict(checkpoint_data['metrics'])
        
        iteration = checkpoint_data['iteration']
        print(f"Loaded checkpoint from iteration {iteration}")
        return iteration
    
    def save_metrics_csv(self):
        """Save metrics to CSV file."""
        csv_path = os.path.join(self.data_dir, "training_metrics.csv")
        metrics_dict = self.metrics.to_dict()
        
        if not metrics_dict['iteration']:
            return
        
        max_len = max(len(m) for m in metrics_dict.values() if isinstance(m, list))
        
        # Pad all lists to same length
        padded_dict = {}
        for key, val in metrics_dict.items():
            if isinstance(val, list):
                padded_dict[key] = val + [None] * (max_len - len(val))
            else:
                padded_dict[key] = val
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=padded_dict.keys())
            writer.writeheader()
            for i in range(max_len):
                row = {k: v[i] if i < len(v) else None for k, v in padded_dict.items()}
                writer.writerow(row)
    
    def compute_batch_metrics(self, batch):
        """Compute loss metrics for a batch."""
        batch_loss = 0.0
        value_loss = 0.0
        policy_loss = 0.0
        count = 0
        
        for state, pi, z in batch:
            x = self.net._state_to_vector(state)
            a, output = self.net._forward(x)
            
            # Value loss
            v = output[66]
            v_err = (v - z) ** 2
            value_loss += v_err
            
            # Policy loss (simplified)
            legal_moves = self.net._get_legal_moves(state)
            if legal_moves:
                pred_policy = self.net._policy_from_output(output, legal_moves)
                for move_idx in legal_moves:
                    move_tuple = tuple(move_idx)
                    true_prob = pi.get(move_tuple, 1e-8)
                    pred_prob = max(pred_policy.get(move_tuple, 1e-8), 1e-10)
                    p_loss = true_prob * (np.log(true_prob + 1e-10) - np.log(pred_prob))
                    policy_loss += p_loss
            
            batch_loss += v_err
            count += 1
        
        if count > 0:
            batch_loss /= count
            value_loss /= count
            policy_loss /= count
        
        return batch_loss, value_loss, policy_loss
    
    def update_visualization(self):
        """Update live plots."""
        self.fig.clear()
        metrics = self.metrics
        
        # Loss plot
        ax = self.fig.add_subplot(2, 3, 1)
        ax.plot(metrics.iteration, metrics.avg_loss, 'b-', label='Avg Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Average Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Value loss
        ax = self.fig.add_subplot(2, 3, 2)
        ax.plot(metrics.iteration, metrics.value_loss, 'g-', label='Value Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Policy loss
        ax = self.fig.add_subplot(2, 3, 3)
        ax.plot(metrics.iteration, metrics.policy_loss, 'r-', label='Policy Loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Game length
        ax = self.fig.add_subplot(2, 3, 4)
        if metrics.game_length:
            ax.plot(range(len(metrics.game_length)), metrics.game_length, 'm-', label='Avg Game Length')
            ax.set_xlabel('Game')
            ax.set_ylabel('Moves')
            ax.set_title('Average Game Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Win rate
        ax = self.fig.add_subplot(2, 3, 5)
        if metrics.win_rate:
            ax.plot(range(len(metrics.win_rate)), metrics.win_rate, 'c-', label='Win Rate')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Game')
            ax.set_ylabel('Win Rate')
            ax.set_title('Win Rate Over Time')
            ax.set_ylim([0, 1])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = self.fig.add_subplot(2, 3, 6)
        ax.plot(metrics.iteration, metrics.learning_rate, 'k-', label='Learning Rate')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Only pause if backend is interactive
        if matplotlib.get_backend().lower() not in ['agg', 'template']:
            plt.pause(0.1)
        else:
            plt.draw()
    
    def train(self, num_iterations=1000, games_per_iteration=10, batch_size=32, 
              initial_lr=1e-3, lr_decay=0.9999):
        """Enhanced training loop with metrics collection and visualization.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Self-play games per iteration
            batch_size: Training batch size
            initial_lr: Starting learning rate
            lr_decay: Learning rate decay per iteration
        """
        current_lr = initial_lr
        start_time = datetime.now()
        game_count = 0
        
        print(f"Starting training: {num_iterations} iterations, {games_per_iteration} games/iter")
        print(f"Data directory: {self.data_dir}")
        print(f"Initial learning rate: {current_lr}")
        
        try:
            for it in range(num_iterations):
                # Generate self-play games
                for g in range(games_per_iteration):
                    state = Boardstate()
                    examples = self.generate_self_play(state, temperature=1.0)
                    
                    # Track game stats
                    game_length = len(examples)
                    result = state.is_won()
                    if result:
                        winner_color, _ = result
                        for ex_state, _, _ in examples:
                            current = ex_state.turn_colour()
                            self.metrics.record_game(game_length, winner_color, current)
                    
                    for ex in examples:
                        self.replay.push(ex)
                    
                    game_count += 1
                
                # Train on accumulated examples
                if len(self.replay) > 0:
                    batch = self.replay.sample(min(batch_size, len(self.replay)))
                    
                    # Compute metrics before training
                    batch_loss, value_loss, policy_loss = self.compute_batch_metrics(batch)
                    self.metrics.record_batch(it, batch_loss, value_loss, policy_loss, current_lr)
                    
                    # Training step
                    self.net.train(batch, lr=current_lr, debug=False)
                
                # Learning rate decay
                current_lr *= lr_decay
                
                # Periodic checkpoint and visualization
                if (it + 1) % self.checkpoint_interval == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"Iteration {it+1}/{num_iterations} - Loss: {self.metrics.avg_loss[-1]:.4f}, "
                          f"Games: {game_count}, Time: {elapsed:.1f}s")
                    self.save_latest_checkpoint(it)
                    self.save_metrics_csv()
                
                if (it + 1) % self.visualize_interval == 0:
                    self.update_visualization()
            
            # Final save
            self.save_checkpoint(num_iterations)
            self.save_metrics_csv()
            self.update_visualization()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"\n{'='*50}")
            print(f"Training completed!")
            print(f"Total iterations: {num_iterations}")
            print(f"Total games: {game_count}")
            print(f"Total time: {elapsed:.1f}s")
            print(f"Final loss: {self.metrics.avg_loss[-1]:.4f}")
            print(f"Checkpoint saved to: {self.data_dir}")
            print(f"{'='*50}\n")
            
            # Keep plots visible (only if interactive backend)
            if matplotlib.get_backend().lower() not in ['agg', 'template']:
                plt.show(block=True)
            else:
                # Save final plot for non-interactive backends
                plot_path = os.path.join(self.data_dir, "training_plots.png")
                self.fig.savefig(plot_path, dpi=100, bbox_inches='tight')
                print(f"Training plots saved to: {plot_path}\n")
            
        except KeyboardInterrupt:
            print("\n\nTraining interrupted! Saving checkpoint...")
            self.save_latest_checkpoint(num_iterations)
            self.save_metrics_csv()
            print(f"Emergency checkpoint saved")


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train MCTS player for Onitama")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=10, help="Games per iteration")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--lr-decay", type=float, default=0.9999, help="Learning rate decay")
    parser.add_argument("--data-dir", type=str, default="MCTS_data", help="Data directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    
    args = parser.parse_args()
    
    # Create network and trainer
    net = NeuralNet(seed=42)
    trainer = EnhancedSelfPlayTrainer(net, mcts_simulations=args.mcts_sims, 
                                      data_dir=args.data_dir)
    
    # Try to resume if requested
    if args.resume:
        latest_checkpoint = os.path.join(args.data_dir, "latest_checkpoint.pkl")
        if os.path.exists(latest_checkpoint):
            trainer.load_checkpoint("latest")
            print("Resumed from checkpoint\n")
    
    # Train
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        batch_size=args.batch_size,
        initial_lr=args.lr,
        lr_decay=args.lr_decay
    )


if __name__ == "__main__":
    main()
