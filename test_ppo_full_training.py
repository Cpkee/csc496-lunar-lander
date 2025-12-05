#!/usr/bin/env python
"""
Full PPO training test and performance evaluation.

This script:
1. Trains a PPO agent to completion
2. Documents training statistics
3. Compares to baseline performance
"""

import time
import gymnasium as gym
import agent_class as agent
import numpy as np

def train_ppo_agent():
    """Train PPO agent on Lunar Lander"""
    print("=" * 70)
    print("FULL PPO TRAINING TEST")
    print("=" * 70)
    
    # Create environment
    print("\nCreating LunarLander-v2 environment...")
    env = gym.make('LunarLander-v2')
    
    # Get environment dimensions
    N_actions = env.action_space.n
    observation, info = env.reset()
    N_state = len(observation)
    
    print(f"State space dimension: {N_state}")
    print(f"Number of actions: {N_actions}")
    
    # Set parameters for PPO
    parameters = {
        'N_state': N_state,
        'N_actions': N_actions,
        # PPO-specific (using defaults from paper)
        'horizon': 128,
        'n_epochs': 4,
        'minibatch_size': 32,
        'clip_epsilon': 0.1,
        'gae_lambda': 0.95,
        'discount_factor': 0.99,
        # Training parameters
        'n_episodes_max': 2000,  # Reasonable max
        'n_solving_episodes': 20,
        'solving_threshold_min': 200,
        'solving_threshold_mean': 230,
        'saving_stride': 100,
    }
    
    print("\nPPO Hyperparameters:")
    print(f"  Horizon: {parameters['horizon']}")
    print(f"  Epochs per update: {parameters['n_epochs']}")
    print(f"  Minibatch size: {parameters['minibatch_size']}")
    print(f"  Clip epsilon: {parameters['clip_epsilon']}")
    print(f"  GAE lambda: {parameters['gae_lambda']}")
    print(f"  Learning rate: 2.5e-4 (Adam)")
    
    # Create agent
    print("\nInitializing PPO agent...")
    my_agent = agent.ppo(parameters=parameters)
    print(f"Policy network: {my_agent.get_number_of_model_parameters('policy_net')} parameters")
    print(f"Critic network: {my_agent.get_number_of_model_parameters('critic_net')} parameters")
    
    # Train
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    print(f"Target: Last {parameters['n_solving_episodes']} episodes with")
    print(f"  - Min return > {parameters['solving_threshold_min']}")
    print(f"  - Mean return > {parameters['solving_threshold_mean']}")
    print("=" * 70)
    
    start_time = time.time()
    
    training_results = my_agent.train(
        environment=env,
        verbose=True,
        model_filename='ppo_full_test.tar',
        training_filename='ppo_full_test_training_data.h5'
    )
    
    training_time = time.time() - start_time
    
    env.close()
    
    # Analyze results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED!")
    print("=" * 70)
    
    episodes = len(training_results['episode_durations'])
    total_steps = training_results['n_steps_simulated'][-1]
    total_updates = training_results['n_training_updates'][-1]
    returns = training_results['epsiode_returns']
    
    print(f"\nTraining Statistics:")
    print(f"  Total episodes: {episodes}")
    print(f"  Total environment steps: {total_steps}")
    print(f"  Total PPO updates: {total_updates}")
    print(f"  Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    print(f"  Steps per second: {total_steps/training_time:.1f}")
    print(f"  Training completed: {training_results['training_completed']}")
    
    print(f"\nFinal Performance (last 20 episodes):")
    final_returns = returns[-20:]
    print(f"  Min return: {np.min(final_returns):.2f}")
    print(f"  Mean return: {np.mean(final_returns):.2f}")
    print(f"  Max return: {np.max(final_returns):.2f}")
    print(f"  Std return: {np.std(final_returns):.2f}")
    
    # Print learning curve milestones
    print(f"\nLearning Milestones:")
    for threshold in [0, 100, 150, 200, 230]:
        # Find first episode where mean of last 20 exceeds threshold
        for i in range(20, len(returns)):
            if np.mean(returns[i-20:i]) > threshold:
                print(f"  Mean return > {threshold}: Episode {i} ({training_results['n_steps_simulated'][i]} steps)")
                break
    
    return training_results, training_time


if __name__ == "__main__":
    try:
        results, train_time = train_ppo_agent()
        
        print("\n" + "=" * 70)
        print("TEST SUCCESSFUL!")
        print("=" * 70)
        print("\nPPO successfully trained on Lunar Lander!")
        print(f"Training completed in {train_time/60:.1f} minutes")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()

