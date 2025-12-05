#!/usr/bin/env python
"""
PPO training with tuned hyperparameters for Lunar Lander.

Based on the first training attempt, we'll adjust:
1. Reduce learning rate for more stable learning
2. Increase horizon for better value estimates
3. Adjust clipping and entropy coefficients
"""

import time
import gymnasium as gym
import agent_class as agent
import numpy as np
import h5py

def analyze_training_data(filename):
    """Load and analyze training data"""
    with h5py.File(filename, 'r') as hf:
        returns = np.array(hf['epsiode_returns'])
        steps = np.array(hf['n_steps_simulated'])
    return returns, steps

def train_ppo_tuned():
    """Train PPO with tuned hyperparameters"""
    print("=" * 70)
    print("PPO TRAINING WITH TUNED HYPERPARAMETERS")
    print("=" * 70)
    
    env = gym.make('LunarLander-v2')
    N_actions = env.action_space.n
    observation, info = env.reset()
    N_state = len(observation)
    
    # Tuned parameters for more stable learning
    parameters = {
        'N_state': N_state,
        'N_actions': N_actions,
        # Adjusted PPO hyperparameters
        'horizon': 256,  # Increased for better value estimates
        'n_epochs': 4,
        'minibatch_size': 64,  # Increased batch size
        'clip_epsilon': 0.2,  # Slightly more permissive clipping
        'gae_lambda': 0.95,
        'discount_factor': 0.99,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        # Network with different architecture
        'neural_networks': {
            'policy_net': {
                'layers': [N_state, 256, 128, N_actions]  # Larger network
            },
            'critic_net': {
                'layers': [N_state, 256, 128, 1]
            }
        },
        'optimizers': {
            'policy_net': {
                'optimizer': 'Adam',
                'optimizer_args': {'lr': 1e-4}  # Reduced learning rate
            },
            'critic_net': {
                'optimizer': 'Adam',
                'optimizer_args': {'lr': 1e-4}
            }
        },
        # Training parameters
        'n_episodes_max': 1500,
        'n_solving_episodes': 20,
        'solving_threshold_min': 200,
        'solving_threshold_mean': 230,
        'saving_stride': 50,
    }
    
    print("\nTuned PPO Hyperparameters:")
    print(f"  Horizon: {parameters['horizon']} (was 128)")
    print(f"  Minibatch size: {parameters['minibatch_size']} (was 32)")
    print(f"  Clip epsilon: {parameters['clip_epsilon']} (was 0.1)")
    print(f"  Learning rate: 1e-4 (was 2.5e-4)")
    print(f"  Network: [8, 256, 128, output] (was [8, 128, 64, output])")
    
    print("\nInitializing PPO agent...")
    my_agent = agent.ppo(parameters=parameters)
    print(f"Policy network: {my_agent.get_number_of_model_parameters('policy_net')} parameters")
    print(f"Critic network: {my_agent.get_number_of_model_parameters('critic_net')} parameters")
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)
    
    start_time = time.time()
    
    training_results = my_agent.train(
        environment=env,
        verbose=True,
        model_filename='ppo_tuned.tar',
        training_filename='ppo_tuned_training_data.h5'
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
    print(f"  Total steps: {total_steps}")
    print(f"  Total updates: {total_updates}")
    print(f"  Training time: {training_time:.1f}s ({training_time/60:.1f}min)")
    print(f"  Training completed: {training_results['training_completed']}")
    
    print(f"\nFinal Performance (last 20 episodes):")
    final_returns = returns[-20:]
    print(f"  Min: {np.min(final_returns):.2f}")
    print(f"  Mean: {np.mean(final_returns):.2f}")
    print(f"  Max: {np.max(final_returns):.2f}")
    
    # Check for milestones
    print(f"\nLearning Progress:")
    for threshold in [0, 100, 150, 200]:
        for i in range(20, len(returns)):
            if np.mean(returns[i-20:i]) > threshold:
                print(f"  Mean > {threshold}: Episode {i}")
                break
        else:
            print(f"  Mean > {threshold}: Not reached")
    
    # Compare to previous run
    print("\n" + "=" * 70)
    print("Comparison to Default Hyperparameters:")
    print("=" * 70)
    try:
        prev_returns, prev_steps = analyze_training_data('ppo_full_test_training_data.h5')
        print(f"Default config final mean: {np.mean(prev_returns[-20:]):.2f}")
        print(f"Tuned config final mean: {np.mean(returns[-20:]):.2f}")
        print(f"Improvement: {np.mean(returns[-20:]) - np.mean(prev_returns[-20:]):.2f}")
    except:
        print("Could not load previous training data for comparison")
    
    return training_results, training_time


if __name__ == "__main__":
    try:
        results, train_time = train_ppo_tuned()
        print("\n" + "=" * 70)
        print("TEST COMPLETED!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


