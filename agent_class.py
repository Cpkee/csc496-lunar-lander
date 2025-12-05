#!/usr/bin/env python

import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
device = torch.device("cpu") 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'
import warnings
from torch.distributions import Categorical

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# NEW CODE FOR PER
class PrioritizedMemory:
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.position = 0
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        transition = self.Transition(*args)
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(max_priority)
        else:
            self.memory[self.position] = transition
            self.priorities[self.position] = max_priority
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return []

        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance-sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = self.Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)


class PPOMemory(object):
    """
    Memory class for PPO algorithm that stores trajectories and computes
    Generalized Advantage Estimation (GAE).
    
    Unlike the standard replay buffer, this memory:
    - Stores complete trajectories with log probabilities and values
    - Computes advantages using GAE
    - Supports multiple epochs of minibatch training
    - Clears after each policy update
    """
    
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        """
        Initialize PPO memory
        
        Args:
            gamma: Discount factor for rewards
            gae_lambda: Lambda parameter for GAE (λ in the paper)
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # These will be computed by compute_gae()
        self.advantages = None
        self.returns = None
    
    def push(self, state, action, reward, log_prob, value, done):
        """
        Add a transition to the trajectory
        
        Args:
            state: Current state (numpy array or tensor)
            action: Action taken (int or tensor)
            reward: Reward received (float or tensor)
            log_prob: Log probability of the action under current policy (tensor)
            value: Value estimate V(s) from critic (tensor)
            done: Whether episode terminated (bool)
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, last_value=0.0):
        """
        Compute Generalized Advantage Estimation (GAE) and returns.
        
        GAE formula (Schulman et al. 2016):
        δ_t = r_t + γ*V(s_{t+1})*(1-done_t) - V(s_t)
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        
        Returns are computed as: R_t = A_t + V(s_t)
        
        Args:
            last_value: Value estimate for the last state (used if trajectory
                       was truncated but not terminated). Default 0.0.
        """
        # Convert lists to tensors for efficient computation
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        values = torch.cat(self.values)  # Already tensors from critic network
        dones = torch.tensor(self.dones, dtype=torch.float32)
        
        # Initialize advantages list
        advantages = []
        gae = 0.0
        
        # Compute GAE backwards from end of trajectory
        # We iterate backwards because each advantage depends on future advantages
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Last step: use last_value as next value
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            # TD residual: δ_t = r_t + γ*V(s_{t+1})*(1-done_t) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            # We compute this recursively: gae = δ_t + (γλ)*gae
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Convert to tensor
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        
        # Normalize advantages (improves training stability)
        # Only normalize if we have more than 1 sample and std > 0
        if len(self.advantages) > 1:
            adv_mean = self.advantages.mean()
            adv_std = self.advantages.std()
            if adv_std > 1e-8:
                self.advantages = (self.advantages - adv_mean) / adv_std
            # If std is too small, just center the advantages
            elif len(self.advantages) > 1:
                self.advantages = self.advantages - adv_mean
        
        # Compute returns: R_t = A_t + V(s_t)
        # These are the target values for training the critic
        self.returns = self.advantages + values
    
    def get_minibatches(self, minibatch_size):
        """
        Generate random minibatches for training.
        
        This method yields minibatches by randomly shuffling the trajectory data
        and splitting it into batches. This is used for multiple epochs of SGD.
        
        Args:
            minibatch_size: Size of each minibatch
            
        Yields:
            Dictionary containing minibatch of:
                - states: Batch of states
                - actions: Batch of actions
                - old_log_probs: Batch of log probabilities under old policy
                - returns: Batch of computed returns (targets for critic)
                - advantages: Batch of computed advantages (for policy update)
        """
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Must call compute_gae() before get_minibatches()")
        
        # Get total number of samples
        n_samples = len(self.states)
        
        # Convert states list to tensor
        # Handle both numpy arrays and tensors
        if isinstance(self.states[0], torch.Tensor):
            states_tensor = torch.stack(self.states)
        else:
            states_tensor = torch.tensor(np.array(self.states), dtype=torch.float32)
        
        # Convert actions to tensor
        if isinstance(self.actions[0], torch.Tensor):
            actions_tensor = torch.cat(self.actions)
        else:
            actions_tensor = torch.tensor(self.actions, dtype=torch.long)
        
        # Log probs are already tensors
        log_probs_tensor = torch.cat(self.log_probs)
        
        # Create random permutation for shuffling
        indices = torch.randperm(n_samples)
        
        # Generate minibatches
        for start_idx in range(0, n_samples, minibatch_size):
            end_idx = min(start_idx + minibatch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield {
                'states': states_tensor[batch_indices],
                'actions': actions_tensor[batch_indices],
                'old_log_probs': log_probs_tensor[batch_indices],
                'returns': self.returns[batch_indices],
                'advantages': self.advantages[batch_indices],
            }
    
    def clear(self):
        """
        Clear all stored trajectory data.
        
        This should be called after each policy update in PPO, as we don't
        maintain a long-term replay buffer like in DQN.
        """
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
    def __len__(self):
        """Return the number of transitions currently stored"""
        return len(self.states)


class neural_network(nn.Module):
    '''
    Feedforward neural network with variable number
    of hidden layers and ReLU nonlinearites
    '''

    def __init__(self,
                layers=[8,64,32,4],# layers[i] = # of neurons at i-th layer
                # layers[0] = input layer
                # layers[-1] = output layer
                dropout=False,
                p_dropout=0.5,
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            #
            self.network_layers.append(nn.Linear(neurons_in_current_layer, 
                                                layers[i+1]) )
            #
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )
            #
            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        #
        self.network_layers = nn.Sequential(*self.network_layers)
        #

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x



class agent_base():

    def __init__(self,parameters):
        """
        Initializes the agent class

        Keyword arguments:
        parameters -- dictionary with parameters for the agent

        There are two mandatory keys for the dictionary:
        - N_state (int): dimensionality of the (continuous) state space
        - N_actions (int): number of actions available to the agent

        All other arguments are optional, for a list see the class methods 
        get_default_parameters(self,parameters)
        set_parameters(self,parameters)

        """
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        # set parameters that are mandatory and can only be set at 
        # initializaton of a class instance
        self.set_initialization_parameters(parameters=parameters)
        #
        # get dictionary with default parameters
        default_parameters = self.get_default_parameters()
        # for all parameters not set by the input dictionary, add the 
        # respective default parameter
        parameters = self.merge_dictionaries(dict1=parameters,
                                             dict2=default_parameters)
        # set all parameters (except for those already set above in 
        # self.set_initialization_parameters())
        self.set_parameters(parameters=parameters)
        #
        # for future reference, each instance of a class carries a copy of 
        # the parameters as internal variable
        self.parameters = copy.deepcopy(parameters)
        #
        # intialize neural networks 
        self.initialize_neural_networks(neural_networks=\
                                            parameters['neural_networks'])
        # initialize the optimizer and loss function used for training
        self.initialize_optimizers(optimizers=parameters['optimizers'])
        self.initialize_losses(losses=parameters['losses'])
        #
        self.in_training = False

    def make_dictionary_keys_lowercase(self,dictionary):
        output_dictionary = {}
        for key, value in dictionary.items():
            output_dictionary[key.lower()] = value
        return output_dictionary

    def merge_dictionaries(self,dict1,dict2):
        '''
        Merge two dictionaries and return the merged dictionary

        If a key "key" exists in both dict1 and dict2, then the value from
        dict1 is used for the returned dictionary
        '''
        #
        return_dict = copy.deepcopy(dict1)
        #
        dict1_keys = return_dict.keys()
        for key, value in dict2.items():
            # we just add those entries from dict2 to dict1
            # that do not already exist in dict1
            if key not in dict1_keys:
                return_dict[key] = value
        #
        return return_dict

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the class
        '''
        #
        parameters = {
            'neural_networks':
                {
                'policy_net':{
                    'layers':[self.n_state,128,32,self.n_actions],
                            }
                },
            'optimizers':
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3}, # learning rate
                            }
                },
            'losses':
                {
                'policy_net':{            
                    'loss':'MSELoss',
                }
                },
            #
            'n_memory':20000,
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            #
            'n_episodes_max':10000,
            'n_solving_episodes':20,
            'solving_threshold_min':200,
            'solving_threshold_mean':230,
            #
            'discount_factor':0.99,
            }
        #
        # in case at some point the above dictionary is edited and an upper
        # case key is added:
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        return parameters


    def set_initialization_parameters(self,parameters):
        '''Set those class parameters that are required at initialization'''
        #
        try: # set mandatory parameter N_state
            self.n_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        #
        try: # set mandatory parameter N_actions
            self.n_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")

    def set_parameters(self,parameters):
        """Set training parameters"""
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        ########################################
        # Discount factor for Bellman equation #
        ########################################
        try: # 
            self.discount_factor = parameters['discount_factor']
        except KeyError:
            pass
        #
        #################################
        # Experience replay memory size #
        #################################
        try: # 
            self.n_memory = int(parameters['n_memory'])
            self.memory = PrioritizedMemory(self.n_memory)
        except KeyError:
            pass
        #
        ###############################
        # Parameters for optimization #
        ###############################
        try: # number of simulation timesteps between optimization steps
            self.training_stride = parameters['training_stride']
        except KeyError:
            pass
        #
        try: # size of mini-batch for each optimization step
            self.batch_size = int(parameters['batch_size'])
        except KeyError:
            pass
        #
        try: # IO during training: every saving_stride episodes, the 
            # current status of the training is saved to disk
            self.saving_stride = parameters['saving_stride']
        except KeyError:
            pass
        #
        ##############################################
        # Parameters for training stopping criterion #
        ##############################################
        try: # maximal number of episodes until the training is stopped 
            # (if stopping criterion is not met before)
            self.n_episodes_max = parameters['n_episodes_max']
        except KeyError:
            pass
        #
        try: # # of the last N_solving episodes that need to fulfill the
            # stopping criterion for minimal and mean episode return
            self.n_solving_episodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        #
        try: # minimal return over last N_solving_episodes
            self.solving_threshold_min = parameters['solving_threshold_min']
        except KeyError:
            pass
        #
        try: # mean return over last N_solving_episodes
            self.solving_threshold_mean = parameters['solving_threshold_mean']
        except KeyError:
            pass
        #

    def get_parameters(self):
        """Return dictionary with parameters of the current agent instance"""

        return self.parameters

    def initialize_neural_networks(self,neural_networks):
        """Initialize all neural networks"""

        self.neural_networks = {}
        for key, value in neural_networks.items():
            self.neural_networks[key] = neural_network(value['layers']).to(device)
        
    def initialize_optimizers(self,optimizers):
        """Initialize optimizers"""

        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neural_networks[key].parameters(),
                            **value['optimizer_args'])
    
    def initialize_losses(self,losses):
        """Instantiate loss functions"""

        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self,name='policy_net'): 
        """Return the number of trainable neural network parameters"""
        # from https://stackoverflow.com/a/49201237
        return sum(p.numel() for p in self.neural_networks[name].parameters() \
                                    if p.requires_grad)


    def get_state(self):
        '''Return dictionary with current state of neural net and optimizer'''
        #
        state = {'parameters':self.get_parameters()}
        #
        for name,neural_network in self.neural_networks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
        #
        for name,optimizer in (self.optimizers).items():
            #
            state[name+'_optimizer'] = copy.deepcopy(optimizer.state_dict())
        #
        return state
    

    def load_state(self,state):
        '''
        Load given states for neural networks and optimizer

        The argument "state" has to be a dictionary with the following 
        (key, value) pairs:

        1. state['parameters'] = dictionary with the agents parameters
        2. For every neural network, there should be a state dictionary:
            state['$name'] = state dictionary of neural_network['$name']
        3. For every optimizer, there should be a state dictionary:
            state['$name_optimizer'] = state dictionary of optimizers['$name']
        '''
        #
        parameters=state['parameters']
        #
        self.check_parameter_dictionary_compatibility(parameters=parameters)
        #
        self.__init__(parameters=parameters)
        #
        #
        for name,state_dict in (state).items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer','')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neural_networks[name].load_state_dict(state_dict)
        #


    def check_parameter_dictionary_compatibility(self,parameters):
        """Check compatibility of provided parameter dictionary with class"""

        error_string = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try: 
            n_state =  parameters['n_state']
            if n_state != self.n_state:
                raise RuntimeError(error_string.format('n_state',n_state,
                                                self.n_state))
        except KeyError:
            pass
        #
        try: 
            n_actions =  parameters['n_actions']
            if n_actions != self.n_actions:
                raise RuntimeError(error_string.format('n_actions',n_actions,
                                                self.n_actions))
        except KeyError:
            pass


    def evaluate_stopping_criterion(self,list_of_returns):
        """ Evaluate stopping criterion """
        # if we have run at least self.N_solving_episodes, check
        # whether the stopping criterion is met
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.
        #
        # get numpy array with recent returns
        recent_returns = np.array(list_of_returns)
        recent_returns = recent_returns[-self.n_solving_episodes:]
        #
        # calculate minimal and mean return over the last
        # self.n_solving_episodes epsiodes 
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)
        #
        # check whether stopping criterion is met
        if minimal_return > self.solving_threshold_min:
            if mean_return > self.solving_threshold_mean:
                return True, minimal_return, mean_return
        # if stopping crtierion is not met:
        return False, minimal_return, mean_return


    def act(self,state):
        """
        Select an action for the current state
        """
        #
        # This typically uses the policy net. See the child classes below
        # for examples:
        # - dqn: makes decisions using an epsilon-greedy algorithm
        # - actor_critic: draws a stochastic decision with probabilities given
        #                 by the current stochastic policy
        #
        # As an example, we here draw a fully random action:
        return np.random.randint(self.n_actions) 


    def add_memory(self,memory):
        """Add current experience tuple to the memory"""
        self.memory.push(*memory)

    def get_samples_from_memory(self):
        '''
        Get a tuple (states, actions, next_states, rewards, episode_end? ) 
        from the memory, as appopriate for experience replay
        '''
        #
        # get random sample of transitions from memory
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        #
        # convert list of Transition elements to Transition element with lists
        # (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*current_transitions))
        #
        # convert lists of current transitions to torch tensors
        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0)#.to(device)
        # state_batch.shape = [batch_size, N_states]
        next_state_batch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action)#.to(device)
        # action_batch.shape = [batch_size]
        reward_batch = torch.cat(batch.reward)#.to(device)
        done_batch = torch.tensor(batch.done).float()#.to(device)
        #
        return state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch


    def run_optimization_step(self, epoch):
        """Run one optimization step
        
        Keyword argument:
        epoch (int) -- number of current training epoch
        """
        #
        # Here is where the actual optimization happens.
        # 
        # This method MUST be implemented in any child class, and might look
        # very different depending on the learning algorithm. 
        # Note that any implementation must contain the argument "epoch", as 
        # this method is called as run_optimization_step(epoch=epoch) in the
        # method self.train() below.
        # 
        # For examples see the child classes "dqn" and "actor_critic" below
        #
        
    

    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                ):
        """
        Train the agent on a provided environment

        Keyword arguments:
        environment -- environment used by the agent to train. This should be
                       an instance of a class with methods "reset" and "step".
                       - environment.reset() should reset the environment to
                         an initial state and return a tuple,
                            current_state, info = environment.reset(),
                         such current_state is an initial state of the with
                         np.shape(current_state) = (self.N_state,)
                       - environment.set(action) should take an integer in 
                         {0, ..., self.N_action-1} and return a tuple, 
                            s, r, te, tr, info = environment.step(action),
                         where s is the next state with shape (self.N_state,),
                         r is the current reward (a float), and where te and
                         tr are two Booleans that tell us whether the episode
                         has terminated (te == True) or has been truncated 
                         (tr == True)
        verbose (Bool) -- Print progress of training to terminal. Defaults to
                          True
        model_filename (string) -- Output filename for final trained model and
                                   periodic snapshots of the model during 
                                   training. Defaults to None, in which case
                                   nothing is not written to disk
        training_filename (string) -- Output filename for training data, 
                                      namely lists of episode durations, 
                                      episode returns, number of training 
                                      epochs, and total number of steps 
                                      simulated. Defaults to None, in which 
                                      case no training data is written to disk
        """
        self.in_training = True
        #
        training_complete = False
        step_counter = 0 # total number of simulated environment steps
        epoch_counter = 0 # number of training epochs 
        #
        # lists for documenting the training
        episode_durations = [] # duration of each training episodes
        episode_returns = [] # return of each training episode
        steps_simulated = [] # total number of steps simulated at the end of
                             # each training episode
        training_epochs = [] # total number of training epochs at the end of 
                             # each training episode
        #
        output_state_dicts = {} # dictionary in which we will save the status
                                # of the neural networks and optimizer
                                # every self.saving_stride steps epochs during
                                # training. 
                                # We also store the final neural network
                                # resulting from our training in this 
                                # dictionary
        #
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            #
            status_progress_string = ( # for outputting status during training
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        #
        for n_episode in range(self.n_episodes_max):
            #
            # reset environment and reward of current episode
            state, info = environment.reset()
            current_total_reward = 0.
            #
            for i in itertools.count(): # timesteps of environment
                #
                # select action using policy net
                action = self.act(state=state)
                #
                # perform action
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)
                #
                step_counter += 1 # increase total steps simulated
                done = terminated or truncated # did the episode end?
                current_total_reward += reward # add current reward to total
                #
                # store the transition in memory
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)
                self.add_memory([torch.tensor(state),
                            action,
                            torch.tensor(next_state),
                            reward,
                            done])
                #
                state = next_state
                #
                if step_counter % self.training_stride == 0:
                    # train model
                    self.run_optimization_step(epoch=epoch_counter) # optimize
                    epoch_counter += 1 # increase count of optimization steps
                #
                if done: # if current episode ended
                    #
                    # update training statistics
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    #
                    # check whether the stopping criterion is met
                    training_complete, min_ret, mean_ret = \
                            self.evaluate_stopping_criterion(\
                                list_of_returns=episode_returns)
                    if verbose:
                            # print training stats
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if min_ret > self.solving_threshold_min:
                                if mean_ret > self.solving_threshold_mean:
                                    end='\n'
                            #
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   min_ret,mean_ret),
                                        end=end)
                    break
            #
            # Save model and training stats to disk
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max-1:
                #
                if model_filename != None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                #
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_returns,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=training_results,
                                        filename=training_filename)
            #
            if training_complete:
                # we stop if the stopping criterion was met at the end of
                # the current episode
                training_results['training_completed'] = True
                break
        #
        if not training_complete:
            # if we stopped the training because the maximal number of
            # episodes was reached, we throw a warning
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.n_episodes_max))
        #
        self.in_training = False
        #
        return training_results

    def save_dictionary(self,dictionary,filename):
        """Save a dictionary in hdf5 format"""

        with h5py.File(filename, 'w') as hf:
            self.save_dictionary_recursively(h5file=hf,
                                            path='/',
                                            dictionary=dictionary)
                
    def save_dictionary_recursively(self,h5file,path,dictionary):
        #
        """
        slightly adapted from https://codereview.stackexchange.com/a/121308
        """
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.save_dictionary_recursively(h5file, 
                                                path + str(key) + '/',
                                                value)
            else:
                h5file[path + str(key)] = value

    def load_dictionary(self,filename):
        with h5py.File(filename, 'r') as hf:
            return self.load_dictionary_recursively(h5file=hf,
                                                    path='/')

    def load_dictionary_recursively(self,h5file, path):
        """
        From https://codereview.stackexchange.com/a/121308
        """
        return_dict = {}
        for key, value in h5file[path].items():
            if isinstance(value, h5py._hl.dataset.Dataset):
                return_dict[key] = value.value
            elif isinstance(value, h5py._hl.group.Group):
                return_dict[key] = self.load_dictionary_recursively(\
                                            h5file=h5file, 
                                            path=path + key + '/')
        return return_dict



class dqn(agent_base):

    def __init__(self,parameters):
        super().__init__(parameters=parameters)
        self.in_training = False

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the dqn
        algorithm
        '''
        #
        default_parameters = super().get_default_parameters()
        #
        # add default parameters specific to the dqn algorithm
        default_parameters['neural_networks']['target_net'] = {}
        default_parameters['neural_networks']['target_net']['layers'] = \
        copy.deepcopy(\
                default_parameters['neural_networks']['policy_net']['layers'])
        #
        #
        # soft update stride for target net:
        default_parameters['target_net_update_stride'] = 1 
        # soft update parameter for target net:
        default_parameters['target_net_update_tau'] = 1e-2 
        #
        # Parameters for epsilon-greedy policy with epoch-dependent epsilon
        default_parameters['epsilon'] = 1.0 # initial value for epsilon
        default_parameters['epsilon_1'] = 0.1 # final value for epsilon
        default_parameters['d_epsilon'] = 0.00005 # decrease of epsilon
            # after each training epoch
        #
        default_parameters['doubledqn'] = True    # activate doubledqn
        #
        return default_parameters


    def set_parameters(self,parameters):
        #
        super().set_parameters(parameters=parameters)
        #
        ##################################################
        # Use deep Q-learning or double deep Q-learning? #
        ##################################################
        try: # False -> use DQN; True -> use double DQN
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        #
        ##########################################
        # Parameters for updating the target net #
        ##########################################
        try: # after how many training epochs do we update the target net?
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        #
        try: # tau for soft update of target net (value 1 means hard update)
            self.target_net_update_tau = parameters['target_net_update_tau']
            # check if provided parameter is within bounds
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        #
        #
        ########################################
        # Parameters for epsilon-greedy policy #
        ########################################
        try: # probability for random action for epsilon-greedy policy
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try: # final probability for random action during training 
            #  for epsilon-greedy policy
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        # 
        try: # amount by which epsilon decreases during each training epoch
            #  until the final value self.epsilon_1 is reached
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):
        """
        Use policy net to select an action for the current state
        
        We use an epsilon-greedy algorithm: 
        - With probability epsilon we take a random action (uniformly drawn
          from the finite number of available actions)
        - With probability 1-epsilon we take the optimal action (as predicted
          by the policy net)

        By default epsilon = 0, which means that we actually use the greedy 
        algorithm for action selection
        """
        #
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            # 
            policy_net = self.neural_networks['policy_net']
            #
            with torch.no_grad():
                policy_net.eval()
                action = policy_net(torch.tensor(state)).argmax(0).item()
                policy_net.train()
                return action
        else:
            # perform random action
            return torch.randint(low=0,high=self.n_actions,size=(1,)).item()
        
    def update_epsilon(self):
        """
        Update epsilon for epsilon-greedy algorithm
        
        For training we assume that 
        epsilon(n) = max{ epsilon_0 - d_epsilon * n ,  epsilon_1 },
        where n is the number of training epochs.

        For epsilon_0 > epsilon_1 the function epsilon(n) is piecewise linear.
        It first decreases from epsilon_0 to epsilon_1 with a slope d_epsilon,
        and then becomes constant at the value epsilon_1.
        
        This ensures that during the initial phase of training the neural 
        network explores more randomly, and in later stages of the training
        follows more the policy learned by the neural net.
        """
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self,epoch):
        """Run one optimization step for the policy net"""
        #
        # if we have less sample transitions than we would draw in an 
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        batch, indices, weights = self.memory.sample(self.batch_size)
        state_batch = torch.cat([torch.tensor(s).unsqueeze(0) for s in batch.state], dim=0)
        next_state_batch = torch.cat([torch.tensor(ns).unsqueeze(0) for ns in batch.next_state], dim=0)
        action_batch = torch.cat([torch.tensor(a) for a in batch.action]).to(device)
        reward_batch = torch.cat([torch.tensor(r, dtype=torch.float32) for r in batch.reward]).to(device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        policy_net = self.neural_networks['policy_net']
        target_net = self.neural_networks['target_net']
        #
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']
        #
        policy_net.train() # turn on training mode
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = policy_net(state_batch.to(device)).gather(dim=1,
                                 index=action_batch.unsqueeze(1))
        # LHS.shape = [batch_size, 1]
        #
        # Evaluate right-hand side of Bellman equation
        if self.doubleDQN:
            # double deep-Q learning paper: https://arxiv.org/abs/1509.06461
            #
            # in double deep Q-learning, we use the policy net for choosing
            # the action on the right-hand side of the Bellman equation. We 
            # then use the target net to evaluate the Q-function on the 
            # chosen action
            argmax_next_state = policy_net(next_state_batch).argmax(
                                                                    dim=1)
            # argmax_next_state.shape = [batch_size]
            #
            Q_next_state = target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
            # shapes of the various tensor appearing in the previous line:
            # self.target_net(next_state_batch).shape = [batch_size,N_actions]
            # self.target_net(next_state_batch).gather(dim=1,
            #   index=argmax_next_state.unsqueeze(1)).shape = [batch_size, 1]
            # Q_next_state.shape = [batch_size]
        else:
            # in deep Q-learning, we use the target net both for choosing
            # the action on the right-hand side of the Bellman equation, and 
            # for evaluating the Q-function on that action
            Q_next_state = target_net(next_state_batch\
                                                ).max(1)[0].detach()
            # Q_next_state.shape = [batch_size]
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        td_errors = (LHS - RHS).squeeze(1)           # shape: [batch_size]
        loss_ = (td_errors.pow(2) * weights).mean()  # # importance-sampling weighted loss
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        #
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities)
        #
        policy_net.eval() # turn off training mode
        #
        self.update_epsilon() # for epsilon-greedy algorithm
        #
        if epoch % self.target_net_update_stride == 0:
            self.soft_update_target_net() # soft update target net
        #
        
    def soft_update_target_net(self):
        """Soft update parameters of target net"""
        #
        # the following code is from https://stackoverflow.com/q/48560227
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)




class actor_critic(agent_base):
    #

    def __init__(self,parameters):

        super().__init__(parameters=parameters)

        #
        self.Softmax = nn.Softmax(dim=0)
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def get_default_parameters(self):
        #
        default_parameters = super().get_default_parameters()
        #
        # add default parameters specific to the dqn algorithm
        default_parameters['neural_networks']['critic_net'] = {}
        default_parameters['neural_networks']['critic_net']['layers'] = \
                    [self.n_state,64,32,1] # needs to have scalar output
        #
        default_parameters['optimizers']['critic_net'] = {
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3}, # learning rate
                            }
        #
        default_parameters['affinities_regularization'] = 0.01
        #
        return default_parameters
    
    def set_parameters(self,parameters):
        #
        super().set_parameters(parameters=parameters)
        #
        try: 
            self.affinities_regularization = \
                            parameters['affinities_regularization']
        except KeyError:
            pass
        #

    def initialize_losses(self,losses):
        """Instantiate loss class
        
        Note that the argument "losses" is mandatory, even though it is not
        used for the particular class
        """

        # for the actor we need a custom loss function
        def loss_actor(state_batch,action_batch,advantage_batch):
            affinities = self.neural_networks['policy_net'](state_batch)
            #
            log_pi_a = self.LogSoftmax(affinities).gather(dim=1,
                                    index=action_batch.unsqueeze(1))
            loss_actor = -log_pi_a * advantage_batch \
                            + self.affinities_regularization \
                                *torch.sum(affinities**2)/self.batch_size
            loss_actor = loss_actor.sum()
            return loss_actor

        self.losses = {}
        self.losses['policy_net'] = loss_actor
        self.losses['critic_net'] = nn.MSELoss()

    def act(self,state):
        """
        Use policy net to select an action for the current state

        For the actor-critic algorithm, the actor chooses an action
        from the available actions
            {1, .., n_action}
        according to their (stochastic) policy.
        More explicitly, for each state s the policy yields a vector of 
        probabilities 
            pi(s) = (pi_1, ..., pi_{n_action})
        for the n_action actions. The actor draws an action according to these
        probabilities pi(s).
        """
        actor_net = self.neural_networks['policy_net']

        with torch.no_grad():
            actor_net.eval()
            # see
            #https://pytorch.org/docs/stable/distributions.html#score-function
            probs = self.Softmax(actor_net(torch.tensor(state)))
            m = Categorical(probs)
            action = m.sample()
            actor_net.train()
            return action.item()
        
    def run_optimization_step(self,epoch):
        """Run one optimization step for the policy net"""
        #
        # Note that the parameter "epoch" is not actually used here, but it is
        # a mandatory parameter because the the method train() in the base 
        # class calls run_optimization_step(epoch=epoch).
        #
        ################################
        # Draw experiences from memory #
        ################################
        # If we have less sample transitions than we would draw in an 
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        state_batch, action_batch, next_state_batch, \
                    reward_batch, done_batch = self.get_samples_from_memory()
        #
        ###################################################################
        # Define local names for neural networks, optimizers, and losses  #
        ###################################################################
        actor_net = self.neural_networks['policy_net']
        critic_net = self.neural_networks['critic_net']
        #
        optimizer_actor = self.optimizers['policy_net']
        optimizer_critic = self.optimizers['critic_net']
        #
        loss_actor = self.losses['policy_net']
        loss_critic = self.losses['critic_net']
        #
        ################
        # train critic #
        ################
        critic_net.train() # turn on training mode
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = critic_net(state_batch.to(device))
        # LHS.shape = [batch_size, 1]
        Q_next_state = critic_net(next_state_batch).detach().squeeze(1)
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        loss = loss_critic(LHS, RHS)
        optimizer_critic.zero_grad()
        loss.backward()
        optimizer_critic.step()
        #
        critic_net.eval() # turn off training mode
        #
        ###############
        # train actor #
        ###############
        actor_net.train()
        advantage_batch = (RHS - LHS).detach()
        loss = loss_actor(state_batch=state_batch,
                          action_batch=action_batch,
                          advantage_batch=advantage_batch)
        optimizer_actor.zero_grad()
        loss.backward()
        optimizer_actor.step()
        #
        actor_net.eval()
        #


class ppo(agent_base):
    """
    Proximal Policy Optimization (PPO) agent implementation.
    
    Based on "Proximal Policy Optimization Algorithms" by Schulman et al. (2017).
    https://arxiv.org/abs/1707.06347
    
    PPO is a policy gradient method that uses:
    1. Clipped surrogate objective to prevent too large policy updates
    2. Generalized Advantage Estimation (GAE) for better advantage estimates
    3. Multiple epochs of minibatch SGD on collected trajectories
    4. Combined loss: policy loss + value loss - entropy bonus
    
    Key differences from Actor-Critic:
    - Uses clipped probability ratio instead of raw log probability
    - Collects trajectories and performs multiple update epochs
    - Uses GAE instead of TD error for advantages
    - Includes entropy bonus for exploration
    """
    
    def __init__(self, parameters):
        """
        Initialize PPO agent
        
        Args:
            parameters: Dictionary with agent parameters including:
                - n_state: Dimensionality of state space
                - n_actions: Number of discrete actions
                - (optional) PPO-specific hyperparameters
        """
        # Call parent initialization first
        # This will call set_parameters() which initializes ppo_memory
        super().__init__(parameters=parameters)
        
        # Initialize PPO-specific components
        self.Softmax = nn.Softmax(dim=1)  # For converting logits to probabilities
        self.LogSoftmax = nn.LogSoftmax(dim=1)  # For computing log probabilities
        
        # Note: ppo_memory is initialized in set_parameters()
    
    def get_default_parameters(self):
        """
        Get default parameters for PPO agent
        
        Returns PPO-specific hyperparameters based on Schulman et al. 2017
        for discrete action spaces (Table 3 in the paper).
        """
        # Get base parameters
        default_parameters = super().get_default_parameters()
        
        # Update network architecture for PPO
        # Actor network (policy_net): outputs action logits
        default_parameters['neural_networks']['policy_net']['layers'] = \
            [self.n_state, 128, 64, self.n_actions]
        
        # Critic network: outputs state value V(s)
        default_parameters['neural_networks']['critic_net'] = {}
        default_parameters['neural_networks']['critic_net']['layers'] = \
            [self.n_state, 128, 64, 1]  # Scalar output for value
        
        # Optimizer for critic network
        default_parameters['optimizers']['critic_net'] = {
            'optimizer': 'Adam',
            'optimizer_args': {'lr': 2.5e-4},  # Learning rate from paper
        }
        
        # Update policy network optimizer to Adam (PPO typically uses Adam)
        default_parameters['optimizers']['policy_net']['optimizer'] = 'Adam'
        default_parameters['optimizers']['policy_net']['optimizer_args'] = {'lr': 2.5e-4}
        
        # PPO-specific hyperparameters (from paper, Table 3 for discrete)
        default_parameters['horizon'] = 128  # T in paper: steps per trajectory
        default_parameters['n_epochs'] = 4  # K in paper: optimization epochs
        default_parameters['minibatch_size'] = 32  # Size of minibatches
        default_parameters['clip_epsilon'] = 0.1  # ε in paper: clipping parameter
        default_parameters['gae_lambda'] = 0.95  # λ in paper: GAE parameter
        default_parameters['value_loss_coef'] = 0.5  # c1 in paper
        default_parameters['entropy_coef'] = 0.01  # c2 in paper
        default_parameters['max_grad_norm'] = 0.5  # Gradient clipping
        
        # Override batch_size to match minibatch_size for consistency
        default_parameters['batch_size'] = 32
        
        # Training stride doesn't apply to PPO (we update after collecting horizon)
        # But keep it for compatibility with base class
        default_parameters['training_stride'] = 128
        
        return default_parameters
    
    def set_parameters(self, parameters):
        """
        Set PPO-specific parameters
        
        Args:
            parameters: Dictionary with agent parameters
        """
        super().set_parameters(parameters=parameters)
        
        # Set default values first in case they're not in parameters
        self.horizon = 128
        self.n_epochs = 4
        self.minibatch_size = 32
        self.clip_epsilon = 0.1
        self.gae_lambda = 0.95
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # PPO-specific parameters (override defaults if provided)
        try:
            self.horizon = parameters['horizon']
        except KeyError:
            pass
        
        try:
            self.n_epochs = parameters['n_epochs']
        except KeyError:
            pass
        
        try:
            self.minibatch_size = parameters['minibatch_size']
        except KeyError:
            pass
        
        try:
            self.clip_epsilon = parameters['clip_epsilon']
        except KeyError:
            pass
        
        try:
            self.gae_lambda = parameters['gae_lambda']
        except KeyError:
            pass
        
        try:
            self.value_loss_coef = parameters['value_loss_coef']
        except KeyError:
            pass
        
        try:
            self.entropy_coef = parameters['entropy_coef']
        except KeyError:
            pass
        
        try:
            self.max_grad_norm = parameters['max_grad_norm']
        except KeyError:
            pass
        
        # Initialize PPO memory with discount factor and GAE lambda
        # This needs to happen after all parameters are set
        self.ppo_memory = PPOMemory(
            gamma=self.discount_factor,
            gae_lambda=self.gae_lambda
        )
    
    def initialize_optimizers(self, optimizers):
        """
        Initialize optimizers for both actor and critic networks
        
        Args:
            optimizers: Dictionary specifying optimizer configurations
        """
        self.optimizers = {}
        for key, value in optimizers.items():
            optimizer_name = value['optimizer']
            optimizer_args = value['optimizer_args']
            
            # Use Adam optimizer (standard for PPO)
            if optimizer_name == 'Adam':
                self.optimizers[key] = torch.optim.Adam(
                    self.neural_networks[key].parameters(),
                    **optimizer_args
                )
            else:
                # Fallback to RMSprop if specified
                self.optimizers[key] = torch.optim.RMSprop(
                    self.neural_networks[key].parameters(),
                    **optimizer_args
                )
    
    def initialize_losses(self, losses):
        """
        Initialize loss functions for PPO
        
        Note: The actual PPO loss will be computed in run_optimization_step()
        as it requires special handling of clipped objectives. This method
        is kept for compatibility with the base class.
        
        Args:
            losses: Dictionary specifying loss configurations (not used directly)
        """
        # Value function uses MSE loss
        self.value_loss_fn = nn.MSELoss()
        
        # Policy loss will be computed using clipped surrogate objective
        # Entropy loss will be computed from action distribution
        # These are implemented in compute_ppo_loss() method
    
    def act(self, state, deterministic=False):
        """
        Select an action using the current policy
        
        For PPO, we use a stochastic policy during training and can optionally
        use a deterministic policy (argmax) during evaluation.
        
        Args:
            state: Current state (numpy array or tensor)
            deterministic: If True, select action with highest probability
                          If False, sample from probability distribution
        
        Returns:
            action (int): Selected action
            log_prob (tensor): Log probability of selected action (only during training)
            value (tensor): Value estimate V(s) from critic (only during training)
        """
        actor_net = self.neural_networks['policy_net']
        critic_net = self.neural_networks['critic_net']
        
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            actor_net.eval()
            critic_net.eval()
            
            # Get action logits from actor
            logits = actor_net(state_tensor)
            
            # Convert to probabilities
            probs = self.Softmax(logits)
            
            # Create categorical distribution
            dist = Categorical(probs)
            
            # Select action
            if deterministic and not self.in_training:
                # Deterministic: take action with highest probability
                action = torch.argmax(probs, dim=1)
            else:
                # Stochastic: sample from distribution
                action = dist.sample()
            
            # If in training mode, also compute log_prob and value
            if self.in_training:
                log_prob = dist.log_prob(action)
                value = critic_net(state_tensor).squeeze(-1)
                
                actor_net.train()
                critic_net.train()
                
                return action.item(), log_prob, value
            else:
                return action.item()
    
    def add_memory(self, state, action, reward, log_prob, value, done):
        """
        Add transition to PPO memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action under current policy
            value: Value estimate V(s) from critic
            done: Whether episode terminated
        """
        self.ppo_memory.push(state, action, reward, log_prob, value, done)
    
    def compute_policy_loss(self, states, actions, old_log_probs, advantages):
        """
        Compute the PPO clipped surrogate objective (policy loss).
        
        This implements Equation 7 from Schulman et al. (2017):
        L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
        
        Where:
        - r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) is the probability ratio
        - Â_t is the advantage estimate
        - ε is the clipping parameter (self.clip_epsilon)
        
        The clipping prevents the policy from changing too much in a single update,
        which is the key innovation of PPO.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions taken [batch_size]
            old_log_probs: Log probabilities under old policy [batch_size]
            advantages: Advantage estimates [batch_size]
        
        Returns:
            policy_loss: Negative of the clipped objective (to minimize)
            ratio: Probability ratios (for logging/debugging)
            entropy: Entropy of action distribution (for exploration bonus)
        """
        # Get current policy logits
        actor_net = self.neural_networks['policy_net']
        logits = actor_net(states)
        
        # Convert to probabilities and create distribution
        probs = self.Softmax(logits)
        dist = Categorical(probs)
        
        # Compute new log probabilities for the actions that were taken
        new_log_probs = dist.log_prob(actions)
        
        # Compute probability ratio: r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        # In log space: r_t(θ) = exp(log π_θ(a|s) - log π_θ_old(a|s))
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Compute the two terms of the clipped objective
        # Term 1: r_t(θ) * Â_t (unclipped)
        surr1 = ratio * advantages
        
        # Term 2: clip(r_t(θ), 1-ε, 1+ε) * Â_t (clipped)
        ratio_clipped = torch.clamp(ratio, 
                                     1.0 - self.clip_epsilon, 
                                     1.0 + self.clip_epsilon)
        surr2 = ratio_clipped * advantages
        
        # PPO objective: take minimum of clipped and unclipped
        # We want to maximize this, so we minimize the negative
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute entropy for exploration bonus
        # Entropy = -Σ p(a) * log(p(a))
        entropy = dist.entropy().mean()
        
        return policy_loss, ratio, entropy
    
    def compute_value_loss(self, states, returns):
        """
        Compute the value function loss.
        
        This is a simple Mean Squared Error between the critic's predictions
        and the computed returns from GAE.
        
        Args:
            states: Batch of states [batch_size, state_dim]
            returns: Target returns computed from GAE [batch_size]
        
        Returns:
            value_loss: MSE loss for the value function
        """
        # Get value predictions from critic
        critic_net = self.neural_networks['critic_net']
        values_pred = critic_net(states).squeeze(-1)
        
        # Compute MSE loss
        value_loss = self.value_loss_fn(values_pred, returns)
        
        return value_loss
    
    def compute_ppo_loss(self, states, actions, old_log_probs, returns, advantages):
        """
        Compute the complete PPO loss function.
        
        This combines three components:
        L = L^CLIP(θ) - c1 * L^VF(θ) + c2 * S[π_θ](s_t)
        
        Where:
        - L^CLIP: Clipped surrogate objective (policy loss)
        - L^VF: Value function loss (MSE)
        - S: Entropy bonus for exploration
        - c1: Value loss coefficient (self.value_loss_coef)
        - c2: Entropy coefficient (self.entropy_coef)
        
        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions [batch_size]
            old_log_probs: Old log probabilities [batch_size]
            returns: Target returns from GAE [batch_size]
            advantages: Advantage estimates [batch_size]
        
        Returns:
            total_loss: Combined loss for backpropagation
            loss_dict: Dictionary with individual loss components for logging
        """
        # Compute policy loss (clipped surrogate objective)
        policy_loss, ratio, entropy = self.compute_policy_loss(
            states, actions, old_log_probs, advantages
        )
        
        # Compute value loss
        value_loss = self.compute_value_loss(states, returns)
        
        # Combine losses with coefficients
        # Note: policy_loss is already negative (we want to maximize objective)
        # value_loss is positive (we want to minimize MSE)
        # entropy is positive (we want to maximize, so subtract)
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
        
        # Create dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item() if len(ratio) > 1 else 0.0,
        }
        
        return total_loss, loss_dict
    
    def run_optimization_step(self):
        """
        Run PPO optimization step on collected trajectory.
        
        This method:
        1. Computes GAE on the collected trajectory
        2. Performs multiple epochs of minibatch SGD
        3. Updates both policy and value networks
        4. Applies gradient clipping
        5. Clears memory after updates
        
        This is called after collecting a full trajectory (horizon steps).
        """
        # Check if we have enough data
        if len(self.ppo_memory) == 0:
            return
        
        # Get the last value for GAE computation
        # This is used for bootstrapping if trajectory was truncated
        with torch.no_grad():
            critic_net = self.neural_networks['critic_net']
            critic_net.eval()
            # Get the last state's value
            if len(self.ppo_memory.states) > 0:
                last_state = self.ppo_memory.states[-1]
                if not isinstance(last_state, torch.Tensor):
                    last_state = torch.tensor(last_state, dtype=torch.float32)
                last_state = last_state.unsqueeze(0)
                last_value = critic_net(last_state).squeeze(-1).item()
                # If last transition was terminal, use 0
                if self.ppo_memory.dones[-1]:
                    last_value = 0.0
            else:
                last_value = 0.0
            critic_net.train()
        
        # Compute GAE advantages and returns
        self.ppo_memory.compute_gae(last_value=last_value)
        
        # Perform multiple epochs of optimization
        for epoch in range(self.n_epochs):
            # Get minibatches for this epoch
            for batch in self.ppo_memory.get_minibatches(self.minibatch_size):
                # Extract batch data
                states = batch['states']
                actions = batch['actions']
                old_log_probs = batch['old_log_probs']
                returns = batch['returns']
                advantages = batch['advantages']
                
                # Set networks to training mode
                self.neural_networks['policy_net'].train()
                self.neural_networks['critic_net'].train()
                
                # Compute PPO loss
                total_loss, loss_dict = self.compute_ppo_loss(
                    states, actions, old_log_probs, returns, advantages
                )
                
                # Zero gradients
                self.optimizers['policy_net'].zero_grad()
                self.optimizers['critic_net'].zero_grad()
                
                # Backpropagate
                total_loss.backward()
                
                # Clip gradients to prevent exploding gradients
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.neural_networks['policy_net'].parameters(),
                        self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.neural_networks['critic_net'].parameters(),
                        self.max_grad_norm
                    )
                
                # Update networks
                self.optimizers['policy_net'].step()
                self.optimizers['critic_net'].step()
        
        # Clear memory after updates
        self.ppo_memory.clear()
    
    def train(self, environment, verbose=True, model_filename=None, training_filename=None):
        """
        Train the PPO agent on a provided environment.
        
        This overrides the base class train() method to implement PPO-specific
        trajectory collection and update pattern.
        
        PPO training loop:
        1. Collect trajectory of horizon steps (or until episode ends)
        2. Compute GAE on collected trajectory
        3. Perform multiple epochs of minibatch SGD
        4. Clear memory and repeat
        
        Args:
            environment: Environment with reset() and step() methods
            verbose: Print training progress (default: True)
            model_filename: Path to save model checkpoints
            training_filename: Path to save training statistics
        
        Returns:
            training_results: Dictionary with training statistics
        """
        self.in_training = True
        
        training_complete = False
        step_counter = 0  # Total environment steps
        update_counter = 0  # Number of PPO updates performed
        
        # Training statistics
        episode_durations = []
        episode_returns = []
        steps_simulated = []
        training_updates = []
        
        output_state_dicts = {}
        
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            
            status_progress_string = (
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        
        # Main training loop
        for n_episode in range(self.n_episodes_max):
            # Reset environment
            state, info = environment.reset()
            current_total_reward = 0.
            episode_steps = 0
            
            # Run episode
            while True:
                # Select action (returns action, log_prob, value during training)
                action, log_prob, value = self.act(state)
                
                # Take action in environment
                next_state, reward, terminated, truncated, info = environment.step(action)
                
                step_counter += 1
                episode_steps += 1
                done = terminated or truncated
                current_total_reward += reward
                
                # Store transition in PPO memory
                self.add_memory(state, action, reward, log_prob, value, done)
                
                state = next_state
                
                # Update policy after collecting horizon steps
                if len(self.ppo_memory) >= self.horizon or done:
                    self.run_optimization_step()
                    update_counter += 1
                
                if done:
                    # Episode ended
                    episode_durations.append(episode_steps)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_updates.append(update_counter)
                    
                    # Check stopping criterion
                    training_complete, min_ret, mean_ret = \
                        self.evaluate_stopping_criterion(episode_returns)
                    
                    if verbose:
                        if n_episode % 100 == 0 and n_episode > 0:
                            end = '\n'
                        else:
                            end = '\r'
                        if min_ret > self.solving_threshold_min:
                            if mean_ret > self.solving_threshold_mean:
                                end = '\n'
                        
                        print(status_progress_string.format(
                            n_episode, current_total_reward, min_ret, mean_ret
                        ), end=end)
                    
                    break
            
            # Save model and training stats periodically
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max - 1:
                
                if model_filename is not None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                
                training_results = {
                    'episode_durations': episode_durations,
                    'epsiode_returns': episode_returns,
                    'n_training_updates': training_updates,
                    'n_steps_simulated': steps_simulated,
                    'training_completed': False,
                }
                
                if training_filename is not None:
                    self.save_dictionary(
                        dictionary=training_results,
                        filename=training_filename
                    )
            
            if training_complete:
                training_results['training_completed'] = True
                break
            
        if not training_complete:
            warning_string = (
                "Warning: Training was stopped because the "
                "maximum number of episodes, {0}, was reached. "
                "But the stopping criterion has not been met."
            )
            warnings.warn(warning_string.format(self.n_episodes_max))
        
        self.in_training = False
        
        return training_results


