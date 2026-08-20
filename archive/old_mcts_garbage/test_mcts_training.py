#!/usr/bin/env python3
"""
Test script for MCTS training pipeline.
Tests that:
1. NeuralNet can convert states to vectors
2. MCTS can search and generate policies
3. Self-play games generate valid training examples
4. Training updates the network
"""

import numpy as np
from LeanBoardstate import Boardstate
from MCTS import NeuralNet, MCTS, SelfPlayTrainer


def test_neural_net():
    """Test NeuralNet state vectorization, forward pass, and prediction."""
    print("\n=== Testing NeuralNet ===")
    
    net = NeuralNet(seed=42)
    state = Boardstate()
    
    # Test state to vector conversion
    print("Testing _state_to_vector...")
    vec = net._state_to_vector(state)
    print(f"  Input vector shape: {vec.shape}, expected: {(net.input_size,)}")
    assert vec.shape == (net.input_size,), f"Vector shape mismatch: {vec.shape} != {(net.input_size,)}"
    
    # Test forward pass
    print("Testing _forward...")
    a, output = net._forward(vec)
    print(f"  Output shape: {output.shape}, expected: {(67,)}")
    assert output.shape == (67,), f"Output shape mismatch: {output.shape} != (67,)"
    
    # Test predict
    print("Testing predict...")
    policy, value = net.predict(state)
    print(f"  Policy type: {type(policy)}, number of moves: {len(policy)}")
    print(f"  Value: {value}, range: [-1, 1]")
    assert isinstance(policy, dict), "Policy should be a dictionary"
    assert -1 <= value <= 1, f"Value out of range: {value}"
    
    # Check that all policy values are non-negative and sum to ~1
    if policy:
        policy_sum = sum(policy.values())
        print(f"  Policy sum: {policy_sum}")
        assert 0.99 <= policy_sum <= 1.01, f"Policy doesn't sum to 1: {policy_sum}"
    
    print("✓ NeuralNet tests passed")
    return net


def test_mcts_search():
    """Test MCTS search functionality."""
    print("\n=== Testing MCTS Search ===")
    
    net = NeuralNet(seed=42)
    mcts = MCTS(net, n_simulations=10)  # Few simulations for fast testing
    state = Boardstate()
    
    print("Testing MCTS.search...")
    policy = mcts.search(state)
    print(f"  Number of moves in policy: {len(policy)}")
    print(f"  Policy sum: {sum(policy.values())}")
    
    assert isinstance(policy, dict), "Policy should be a dictionary"
    assert len(policy) > 0, "Policy should have at least one move"
    assert 0.99 <= sum(policy.values()) <= 1.01, "Policy should sum to 1"
    
    print("✓ MCTS search tests passed")


def test_self_play():
    """Test self-play game generation."""
    print("\n=== Testing Self-Play ===")
    
    net = NeuralNet(seed=42)
    mcts = MCTS(net, n_simulations=5)  # Few simulations for fast testing
    trainer = SelfPlayTrainer(net, mcts_simulations=5)
    
    state = Boardstate()
    print("Generating self-play game...")
    examples = trainer.generate_self_play(state, temperature=1.0)
    
    print(f"  Number of moves/examples: {len(examples)}")
    assert len(examples) > 0, "Should generate at least one example"
    
    # Check example format
    for i, (ex_state, ex_pi, ex_z) in enumerate(examples):
        assert ex_state is not None, f"Example {i}: state is None"
        assert isinstance(ex_pi, dict), f"Example {i}: pi should be dict"
        assert isinstance(ex_z, float), f"Example {i}: z should be float"
        assert -1 <= ex_z <= 1, f"Example {i}: z out of range [{ex_z}]"
        assert len(ex_pi) > 0, f"Example {i}: pi is empty"
        assert 0.99 <= sum(ex_pi.values()) <= 1.01, f"Example {i}: pi doesn't sum to 1"
    
    print(f"  All {len(examples)} examples are valid")
    print(f"  Final game outcome: z = {examples[-1][2]}")
    print("✓ Self-play tests passed")


def test_training():
    """Test training on generated examples."""
    print("\n=== Testing Training ===")
    
    net = NeuralNet(seed=42)
    trainer = SelfPlayTrainer(net, mcts_simulations=5, replay_size=1000)
    
    # Generate examples
    print("Generating training examples...")
    for _ in range(2):
        state = Boardstate()
        examples = trainer.generate_self_play(state, temperature=1.0)
        for ex in examples:
            trainer.replay.push(ex)
    
    print(f"  Replay buffer size: {len(trainer.replay)}")
    assert len(trainer.replay) > 0, "Replay buffer should have examples"
    
    # Train
    print("Running training step...")
    batch = trainer.replay.sample(min(8, len(trainer.replay)))
    print(f"  Batch size: {len(batch)}")
    
    # Store initial weights and biases
    initial_weight_66 = net.W_out[66, 0].copy()
    initial_bias_66 = net.b_out[66].copy()
    
    # Debug: print sample batch info
    for i, (state, pi, z) in enumerate(batch[:2]):
        print(f"  Example {i}: pi_len={len(pi)}, z={z}")
    
    # Train with debug
    net.train(batch, lr=1e-1, debug=True)  # Increased learning rate
    
    # Check weights changed (focus on node 66 which has clearer gradients)
    final_weight_66 = net.W_out[66, 0]
    final_bias_66 = net.b_out[66]
    weight_change = abs(final_weight_66 - initial_weight_66)
    bias_change = abs(final_bias_66 - initial_bias_66)
    print(f"  Weight change in W_out[66,0]: {weight_change}")
    print(f"  Bias change in b_out[66]: {bias_change}")
    
    assert weight_change > 1e-9 or bias_change > 1e-9, "Weights or biases should change during training"
    
    print("✓ Training tests passed")


def test_full_pipeline():
    """Test the full training pipeline with multiple iterations."""
    print("\n=== Testing Full Pipeline ===")
    
    net = NeuralNet(seed=42)
    trainer = SelfPlayTrainer(net, mcts_simulations=5, replay_size=100)
    
    print("Running 2 training iterations with 2 games each...")
    trainer.train(num_iterations=2, games_per_iteration=2, batch_size=8)
    
    print(f"  Final replay buffer size: {len(trainer.replay)}")
    print("✓ Full pipeline tests passed")


if __name__ == "__main__":
    print("Starting MCTS training tests...")
    
    try:
        test_neural_net()
        test_mcts_search()
        test_self_play()
        test_training()
        test_full_pipeline()
        
        print("\n" + "="*50)
        print("✓ All tests passed!")
        print("="*50)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
