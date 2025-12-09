# Diagnostic Tools Fixes for PPO Performance Analysis

## Issues Found

The diagnostic tools in `train and visualize agent.ipynb` have several compatibility issues with the actual PPO implementation in `agent_class.py`:

### 1. **Incorrect Parameter Structure**
The notebook uses simplified parameter names that don't match the actual implementation:
- ❌ `'gamma': 0.99` → ✅ `'discount_factor': 0.99`
- ❌ `'layers_actor': [...]` → ✅ `'neural_networks': {'policy_net': {'layers': [...]}}`
- ❌ `'learning_rate_actor': 1e-4` → ✅ `'optimizers': {'policy_net': {'optimizer_args': {'lr': 1e-4}}}`

### 2. **Incorrect Attribute Names**
The diagnostic code references non-existent attributes:
- ❌ `diagnostic_agent.actor` → ✅ `diagnostic_agent.neural_networks['policy_net']`
- ❌ `diagnostic_agent.critic` → ✅ `diagnostic_agent.neural_networks['critic_net']`

### 3. **Incorrect train() Method Call**
The notebook calls `train()` with `N_episodes` parameter, but the actual method doesn't accept it:
- ❌ `train(environment=env, N_episodes=500)` → ✅ `train(environment=env)` (uses `n_episodes_max` from parameters)

### 4. **Memory Access Issues**
The diagnostic code tries to access memory data before `compute_gae()` is called, and the memory is cleared after optimization.

## Fixed Code

Here's the corrected `create_diagnostic_agent` function:

```python
def create_diagnostic_agent(N_state, N_actions, **kwargs):
    """
    Create a PPO agent that tracks detailed training diagnostics
    """
    # Default parameters - using correct parameter structure
    params = {
        'N_state': N_state,
        'N_actions': N_actions,
        'horizon': 256,
        'n_epochs': 4,
        'minibatch_size': 64,
        'clip_epsilon': 0.2,
        'gae_lambda': 0.95,
        'discount_factor': 0.99,  # Note: use 'discount_factor' not 'gamma'
        'neural_networks': {
            'policy_net': {
                'layers': [N_state, 256, 128, N_actions]
            },
            'critic_net': {
                'layers': [N_state, 256, 128, 1]
            }
        },
        'optimizers': {
            'policy_net': {
                'optimizer': 'Adam',
                'optimizer_args': {'lr': 1e-4}
            },
            'critic_net': {
                'optimizer': 'Adam',
                'optimizer_args': {'lr': 1e-4}
            }
        },
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'n_episodes_max': 2000,  # Add this for training length
    }
    
    # Update with any provided kwargs (handle old parameter names for compatibility)
    if 'layers_actor' in kwargs:
        params['neural_networks']['policy_net']['layers'] = kwargs.pop('layers_actor')
    if 'layers_critic' in kwargs:
        params['neural_networks']['critic_net']['layers'] = kwargs.pop('layers_critic')
    if 'learning_rate_actor' in kwargs:
        params['optimizers']['policy_net']['optimizer_args']['lr'] = kwargs.pop('learning_rate_actor')
    if 'learning_rate_critic' in kwargs:
        params['optimizers']['critic_net']['optimizer_args']['lr'] = kwargs.pop('learning_rate_critic')
    if 'gamma' in kwargs:
        params['discount_factor'] = kwargs.pop('gamma')
    
    params.update(kwargs)
    
    # Create agent
    diagnostic_agent = agent.ppo(parameters=params)
    
    # Add diagnostic tracking lists
    diagnostic_agent.diagnostics = {
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'kl_divergences': [],
        'clip_fractions': [],
        'explained_variances': [],
        'ratio_means': [],
        'ratio_stds': [],
    }
    
    # Patch compute_ppo_loss to capture diagnostics (better approach)
    original_compute_loss = diagnostic_agent.compute_ppo_loss
    
    def tracked_compute_loss(states, actions, old_log_probs, returns, advantages):
        """Wrapper to track loss components"""
        total_loss, loss_dict = original_compute_loss(states, actions, old_log_probs, returns, advantages)
        
        # Store diagnostics from loss_dict
        diagnostic_agent.diagnostics['policy_losses'].append(loss_dict.get('policy_loss', 0))
        diagnostic_agent.diagnostics['value_losses'].append(loss_dict.get('value_loss', 0))
        diagnostic_agent.diagnostics['entropies'].append(loss_dict.get('entropy', 0))
        diagnostic_agent.diagnostics['ratio_means'].append(loss_dict.get('ratio_mean', 1.0))
        diagnostic_agent.diagnostics['ratio_stds'].append(loss_dict.get('ratio_std', 0.0))
        
        # Compute additional diagnostics
        with torch.no_grad():
            # Get new policy
            policy_net = diagnostic_agent.neural_networks['policy_net']
            logits = policy_net(states)
            probs = diagnostic_agent.Softmax(logits)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            
            # KL divergence approximation
            kl_div = (old_log_probs - new_log_probs).mean().item()
            diagnostic_agent.diagnostics['kl_divergences'].append(kl_div)
            
            # Clip fraction
            ratios = torch.exp(new_log_probs - old_log_probs)
            clip_frac = ((ratios - 1.0).abs() > diagnostic_agent.clip_epsilon).float().mean().item()
            diagnostic_agent.diagnostics['clip_fractions'].append(clip_frac)
            
            # Explained variance
            critic_net = diagnostic_agent.neural_networks['critic_net']
            new_values = critic_net(states).squeeze(-1)
            var_returns = torch.var(returns)
            explained_var = 1 - torch.var(returns - new_values) / (var_returns + 1e-8)
            explained_var = explained_var.item()
            diagnostic_agent.diagnostics['explained_variances'].append(explained_var)
        
        return total_loss, loss_dict
    
    diagnostic_agent.compute_ppo_loss = tracked_compute_loss
    
    return diagnostic_agent
```

## Fixed Usage Example

```python
# Create diagnostic agent with correct parameters
diagnostic_agent = create_diagnostic_agent(
    N_state=N_state,
    N_actions=N_actions,
    horizon=256,
    n_epochs=4,
    minibatch_size=64,
    clip_epsilon=0.2,
    n_episodes_max=500,  # Set training length here
)

# Train (no N_episodes parameter needed)
env_diag = gym.make('LunarLander-v2')
diagnostic_training_results = diagnostic_agent.train(
    environment=env_diag,
    verbose=True
)
env_diag.close()
```

## Summary

The main fixes needed are:
1. Use correct parameter structure (`neural_networks`, `optimizers`, `discount_factor`)
2. Use correct attribute names (`neural_networks['policy_net']` instead of `actor`)
3. Remove `N_episodes` parameter from `train()` call, set `n_episodes_max` in parameters instead
4. Track diagnostics from `compute_ppo_loss` instead of trying to access cleared memory

