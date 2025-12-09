# PPO Implementation Details

## Overview

This document describes the Proximal Policy Optimization (PPO) implementation in `agent_class.py`, based on the paper "Proximal Policy Optimization Algorithms" by Schulman et al. (2017).

**Key Features:**
- Clipped surrogate objective to prevent large policy updates
- Generalized Advantage Estimation (GAE) for better advantage estimates
- Multiple epochs of minibatch SGD on collected trajectories
- Combined loss: policy loss + value loss - entropy bonus

---

## Architecture Components

### 1. PPOMemory Class

The `PPOMemory` class stores complete trajectories and computes advantages using GAE.

#### Data Storage
```python
self.states = []      # State observations
self.actions = []     # Actions taken
self.rewards = []     # Rewards received
self.log_probs = []   # Log probabilities under old policy
self.values = []      # Value estimates V(s) from critic
self.dones = []       # Episode termination flags
```

#### Key Methods

**`push(state, action, reward, log_prob, value, done)`**
- Adds a single transition to the trajectory
- Stores all necessary information for later GAE computation

**`compute_gae(last_value=0.0)`**
- Computes Generalized Advantage Estimation (GAE) and returns
- **Mathematical Formulation:**
  ```
  TD Residual: δ_t = r_t + γ*V(s_{t+1})*(1-done_t) - V(s_t)
  
  GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
  
  Returns: R_t = A_t + V(s_t)
  ```
- **Implementation Details:**
  - Iterates backwards through trajectory (advantages depend on future)
  - Normalizes advantages: `(A - mean(A)) / std(A)` for stability
  - Computes returns as `A_t + V(s_t)` for critic targets

**`get_minibatches(minibatch_size)`**
- Generates random minibatches for multiple epochs of training
- Shuffles data and splits into batches
- Yields dictionaries with: `states`, `actions`, `old_log_probs`, `returns`, `advantages`

**`clear()`**
- Clears all stored data after policy update
- PPO doesn't maintain a replay buffer like DQN

---

### 2. PPO Agent Class

The `ppo` class inherits from `agent_base` and implements the PPO algorithm.

#### Network Architecture

**Policy Network (`policy_net`):**
- Input: State vector (8 dimensions for LunarLander)
- Output: Action logits (4 actions for LunarLander)
- Default architecture: `[8, 128, 64, 4]`
- Uses Softmax to convert logits to probabilities

**Critic Network (`critic_net`):**
- Input: State vector (8 dimensions)
- Output: Scalar value estimate V(s)
- Default architecture: `[8, 128, 64, 1]`

#### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `horizon` | 128 | Steps per trajectory before update |
| `n_epochs` | 4 | Number of optimization epochs per update |
| `minibatch_size` | 32 | Size of minibatches for SGD |
| `clip_epsilon` | 0.1 | Clipping parameter ε |
| `gae_lambda` | 0.95 | GAE lambda parameter λ |
| `discount_factor` | 0.99 | Discount factor γ |
| `value_loss_coef` | 0.5 | Value loss coefficient c1 |
| `entropy_coef` | 0.01 | Entropy coefficient c2 |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |

---

## Core Methods

### 1. `act(state, deterministic=False)`

**Purpose:** Select an action using the current policy

**Process:**
1. Convert state to tensor
2. Get action logits from policy network
3. Convert to probabilities using Softmax
4. Create Categorical distribution
5. Sample action (or use argmax if deterministic)
6. Compute log_prob and value estimate (during training)

**Returns:**
- During training: `(action, log_prob, value)`
- During evaluation: `action`

**Code Location:** ```1516:1573:agent_class.py```

---

### 2. `compute_policy_loss(states, actions, old_log_probs, advantages)`

**Purpose:** Compute the PPO clipped surrogate objective

**Mathematical Formulation:**
```
Probability Ratio: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
                    = exp(log π_θ(a_t|s_t) - log π_θ_old(a_t|s_t))

Clipped Objective: L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

Where:
- r_t(θ): Probability ratio
- Â_t: Advantage estimate
- ε: Clipping parameter (clip_epsilon)
```

**Implementation:**
1. Get current policy logits
2. Compute new log probabilities
3. Calculate ratio: `exp(new_log_probs - old_log_probs)`
4. Compute unclipped term: `ratio * advantages`
5. Compute clipped term: `clip(ratio, 1-ε, 1+ε) * advantages`
6. Take minimum: `min(unclipped, clipped)`
7. Negate for minimization: `-mean(min(...))`
8. Compute entropy for exploration bonus

**Returns:** `(policy_loss, ratio, entropy)`

**Code Location:** ```1589:1648:agent_class.py```

---

### 3. `compute_value_loss(states, returns)`

**Purpose:** Compute value function loss

**Mathematical Formulation:**
```
L^VF(θ) = MSE(V_θ(s_t), R_t)

Where:
- V_θ(s_t): Critic's value prediction
- R_t: Target return from GAE
```

**Implementation:**
1. Get value predictions from critic network
2. Compute MSE loss between predictions and returns

**Returns:** `value_loss`

**Code Location:** ```1650:1671:agent_class.py```

---

### 4. `compute_ppo_loss(states, actions, old_log_probs, returns, advantages)`

**Purpose:** Compute complete PPO loss combining all components

**Mathematical Formulation:**
```
L(θ) = L^CLIP(θ) - c1 * L^VF(θ) + c2 * S[π_θ](s_t)

Where:
- L^CLIP: Clipped surrogate objective (policy loss)
- L^VF: Value function loss (MSE)
- S: Entropy bonus for exploration
- c1: Value loss coefficient (value_loss_coef = 0.5)
- c2: Entropy coefficient (entropy_coef = 0.01)
```

**Note:** 
- Policy loss is already negative (we maximize objective by minimizing negative)
- Value loss is positive (we minimize MSE)
- Entropy is positive (we maximize, so subtract)

**Returns:** `(total_loss, loss_dict)`

**Code Location:** ```1673:1722:agent_class.py```

---

### 5. `run_optimization_step()`

**Purpose:** Perform PPO optimization on collected trajectory

**Process:**
1. **Check memory:** Return if no data collected
2. **Get last value:** Compute value of last state for GAE bootstrapping
3. **Compute GAE:** Call `ppo_memory.compute_gae(last_value)`
4. **Multiple epochs:** For each epoch:
   - Get random minibatches
   - For each minibatch:
     - Compute PPO loss
     - Zero gradients
     - Backpropagate
     - Clip gradients (if `max_grad_norm` set)
     - Update both networks
5. **Clear memory:** Reset trajectory data

**Key Points:**
- Multiple epochs allow efficient use of collected data
- Random minibatches prevent overfitting to specific transitions
- Gradient clipping prevents exploding gradients
- Memory is cleared after updates (no replay buffer)

**Code Location:** ```1724:1806:agent_class.py```

---

### 6. `train(environment, verbose=True, model_filename=None, training_filename=None)`

**Purpose:** Main training loop for PPO

**Training Flow:**

```
For each episode:
    1. Reset environment
    2. While episode not done:
        a. Select action using act() → (action, log_prob, value)
        b. Take action in environment
        c. Store transition: add_memory(state, action, reward, log_prob, value, done)
        d. If memory >= horizon OR episode done:
           - Call run_optimization_step()
           - Increment update counter
    3. Record episode statistics
    4. Check stopping criterion
    5. Save model periodically
```

**Key Features:**
- Collects trajectories of `horizon` steps (or until episode ends)
- Updates policy after collecting enough data
- Tracks: episode returns, durations, steps, updates
- Supports early stopping based on performance thresholds

**Code Location:** ```1808:1951:agent_class.py```

---

## Mathematical Formulations

### 1. Generalized Advantage Estimation (GAE)

**TD Residual:**
```
δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
```

**GAE (recursive form):**
```
A_t = δ_t + (γλ) * (1 - done_t) * A_{t+1}
```

**Returns:**
```
R_t = A_t + V(s_t)
```

**Why GAE?**
- Reduces variance compared to Monte Carlo returns
- Reduces bias compared to TD(0) estimates
- λ parameter controls bias-variance tradeoff (λ=1 → MC, λ=0 → TD(0))

---

### 2. PPO Clipped Surrogate Objective

**Probability Ratio:**
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**Clipped Objective:**
```
L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

**Why Clipping?**
- Prevents large policy updates that could destabilize training
- Clips ratio to [1-ε, 1+ε] range
- Takes minimum to ensure pessimistic updates

**Visual Explanation:**
- If advantage > 0: increase probability, but cap at (1+ε) ratio
- If advantage < 0: decrease probability, but cap at (1-ε) ratio

---

### 3. Complete PPO Loss

```
L(θ) = L^CLIP(θ) - c1 * L^VF(θ) + c2 * S[π_θ](s_t)
```

**Components:**
1. **Policy Loss (L^CLIP):** Maximize expected advantage (clipped)
2. **Value Loss (L^VF):** Minimize prediction error for returns
3. **Entropy Bonus (S):** Encourage exploration

**Coefficients:**
- `c1 = 0.5`: Weight for value loss
- `c2 = 0.01`: Weight for entropy bonus

---

## Training Algorithm Flow

```
Initialize:
  - Policy network π_θ
  - Critic network V_θ
  - Optimizers (Adam, lr=2.5e-4)
  - PPO memory buffer

For episode = 1 to max_episodes:
    Reset environment, state = s_0
    
    While episode not done:
        # Collect trajectory
        action, log_prob, value = act(state)
        next_state, reward, done = env.step(action)
        add_memory(state, action, reward, log_prob, value, done)
        state = next_state
        
        # Update if enough data collected
        if len(memory) >= horizon OR done:
            run_optimization_step():
                1. Compute GAE advantages
                2. For epoch = 1 to n_epochs:
                    For each minibatch:
                        - Compute PPO loss
                        - Backpropagate
                        - Update networks
                3. Clear memory
    
    Record episode statistics
    Check stopping criterion
```

---

## Key Implementation Details

### 1. Advantage Normalization

Advantages are normalized to have zero mean and unit variance:
```python
advantages = (advantages - mean(advantages)) / std(advantages)
```

**Why?**
- Stabilizes training
- Prevents advantages from becoming too large
- Helps with gradient scaling

### 2. Gradient Clipping

Gradients are clipped to prevent exploding gradients:
```python
torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm=0.5)
```

### 3. Multiple Epochs

PPO performs multiple epochs (default: 4) on the same trajectory:
- More efficient use of collected data
- Prevents overfitting with random minibatches
- Each epoch uses different random shuffling

### 4. No Replay Buffer

Unlike DQN, PPO doesn't maintain a replay buffer:
- Uses on-policy data only
- Clears memory after each update
- Ensures policy and data distribution match

### 5. Stochastic vs Deterministic Actions

- **Training:** Stochastic (sample from distribution)
- **Evaluation:** Can use deterministic (argmax) for better performance

---

## Hyperparameter Tuning Guidelines

### Learning Rate
- Default: 2.5e-4 (Adam)
- Too high: Unstable training, large policy changes
- Too low: Slow learning

### Clip Epsilon (ε)
- Default: 0.1
- Controls how much policy can change per update
- Larger: More aggressive updates, less stable
- Smaller: More conservative, slower learning

### Horizon
- Default: 128 steps
- Number of steps before updating
- Larger: More data per update, but less frequent updates
- Smaller: More frequent updates, but less data per update

### GAE Lambda (λ)
- Default: 0.95
- Controls bias-variance tradeoff
- λ=1: Monte Carlo (low bias, high variance)
- λ=0: TD(0) (high bias, low variance)

### Entropy Coefficient
- Default: 0.01
- Encourages exploration
- Higher: More exploration, slower convergence
- Lower: Less exploration, may get stuck

---

## Differences from Other Algorithms

### vs Actor-Critic
- **PPO:** Uses clipped probability ratio, GAE, multiple epochs
- **AC:** Uses raw log probabilities, TD error, single update

### vs DQN
- **PPO:** On-policy, policy gradient, continuous/discrete actions
- **DQN:** Off-policy, value-based, discrete actions only

### vs REINFORCE
- **PPO:** Uses value function (critic), GAE, clipping
- **REINFORCE:** Pure policy gradient, high variance

---

## Performance Characteristics

**Advantages:**
- Sample efficient (multiple epochs per trajectory)
- Stable training (clipping prevents large updates)
- Works well with continuous and discrete actions
- Simple to implement and tune

**Disadvantages:**
- On-policy (can't reuse old data)
- Requires careful hyperparameter tuning
- May need many environment steps for complex tasks

---

## Code References

- **PPOMemory class:** ```84:270:agent_class.py```
- **PPO agent class:** ```1317:1952:agent_class.py```
- **Policy loss computation:** ```1589:1648:agent_class.py```
- **Value loss computation:** ```1650:1671:agent_class.py```
- **Optimization step:** ```1724:1806:agent_class.py```
- **Training loop:** ```1808:1951:agent_class.py```

---

## References

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). 
"Proximal Policy Optimization Algorithms." 
arXiv preprint arXiv:1707.06347.


