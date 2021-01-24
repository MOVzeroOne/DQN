import gym 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import torch.nn.init as init
from collections import deque
import numpy as np
from itertools import chain
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import wrapper #openAI baseline wrapper
from torch.utils.tensorboard import SummaryWriter
import copy

"""
noisy networks, dueling DQN, double dqn 
"""

class experience_buffer():
    """
    Stores (state, next_state, action, reward, done) tuples in a cyclic memory buffer
    and allows for easy sampling 
    """
    def __init__(self,maxlen:int):
        self.buffer = deque(maxlen=maxlen)
    
    def append(self,experience:tuple) -> None:
        """
        experience: (state, next_state, action, reward, done)
        """
        self.buffer.append(experience)
    
    def sample(self,batch_size:int) -> tuple:
        """
        input:
        returns: (states, next_states, actions, rewards, dones)
        Where the size of each element of the tuple is equal to the batchsize
        """
        indexes = np.random.choice(np.arange(0,len(self.buffer)),batch_size)
        states, next_states, actions, rewards, dones = list(zip(*[self.buffer[index] for index in indexes]))
        
        return (observation_converter(states), observation_converter(next_states), torch.tensor(actions,dtype=torch.long).cuda(), torch.tensor(rewards,dtype=torch.float).cuda(), torch.tensor(dones,dtype=torch.bool).cuda())
    
    def __len__(self) -> int:
        """
        returns the length of the experience buffer
        """
        return len(self.buffer)


class network_conv(nn.Module):
    def __init__(self,in_channels=4,input_shape=(4, 84, 84),amount_actions=6,hidden_size=256):
        super().__init__()
        """
        https://arxiv.org/abs/1312.5602
        We now describe the exact architecture used for all seven Atari games. The input to the neural
        network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8
        filters with stride 4 with the input image and applies a rectifier nonlinearity [10, 18]. The second
        hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
        final hidden layer is fully-connected and consists of 256 rectifier units
        """
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
            mish(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            mish(),
        )         
        conv_out_size = self.get_conv_out(input_shape)
        self.body = nn.Sequential(noisy_layer(conv_out_size,hidden_size),mish()) #noisy
        self.head_advantage = nn.Sequential(nn.Linear(hidden_size,amount_actions))
        self.head_value = nn.Sequential(noisy_layer(hidden_size,1)) 
    
    def get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def resample_noise(self) -> None:
        """
        resample noise for entire network
        """
        for layer in self.body:
            if(layer._get_name() == "noisy_layer"):
                layer.resample_noise()
        
        for layer in self.head_advantage:
            if(layer._get_name() == "noisy_layer"):
                layer.resample_noise()
        
        for layer in self.head_value:
            if(layer._get_name() == "noisy_layer"):
                layer.resample_noise()
        

    def forward(self,x) -> torch.FloatTensor:
        """
        Dueling Network Architectures for Deep Reinforcement Learning
        https://arxiv.org/abs/1511.06581
        the head of the network is seperated out in a value head and an Advantage head.
        (that are recombined together as Q = V + A)  
        trick: 
        Q = V + A - A.mean()
        alternative 
        Q = V +  A - A.max()
        """
        hidden = self.body(self.conv(x).reshape(x.size()[0],-1))
        value = self.head_value(hidden)
        advantage = self.head_advantage(hidden)
        advantage = advantage - advantage.mean(dim=1).view(-1,1)
        
        return value + advantage #Q value 


def observation_converter(obs) -> torch.FloatTensor:
    """
    turns observation in something usable
    """
    return torch.from_numpy(np.array(obs)).view(-1, 84, 84, 4).permute(0,3,1,2).type(torch.float).cuda()/255.0


class dqn_agent():
    def __init__(self,network,env,optimizer,batch_size=64,cyclic_experience_len=400000,gamma=0.99,average=10,steps_before_training=10000):

        self.writer = SummaryWriter()

        self.replay_memory = experience_buffer(cyclic_experience_len) 
        
        self.gamma = torch.tensor(gamma,dtype=torch.float)
        
        self.env = env
        self.optimizer = optimizer
        self.online_network = network.cuda()
        self.target_network = copy.deepcopy(network).cuda()
        
        self.current_observation = self.env.reset() #first observation
        self.batch_size = batch_size
    
        self.steps_before_training = steps_before_training
        self.average = average
        self.steps = 0

        self.highest_mean_reward = 0


    def test_agent(self) -> None:
        """
        tests the performance of the agent and notes it down on tensorboard
        """
        Q_mean = []
        all_total_rewards = []
        
        for i in range(self.average):
            self.current_observation = self.env.reset()
            total_reward = 0
            Q_values = []

            while(True):
                state = self.current_observation
                self.online_network.resample_noise() 
                action,Q = self.choose_action(state,return_Q=True)
                Q_values.append(Q)
                next_state,reward, done,_ = env.step(action)
                
                self.current_observation = next_state #update current observation
                self.replay_memory.append((state, next_state, action, reward, done)) #put information in memory
                total_reward += reward
                if(done):
                    break
            all_total_rewards.append(total_reward)
            Q_mean.append(np.mean(np.array(Q_values)))
        Q_mean = np.array(Q_mean)
        Q_mean_mean = np.mean(Q_mean)
        Q_mean_max = np.max(Q_mean)
        Q_mean_min = np.min(Q_mean)
        Q_mean_std = np.std(Q_mean)
        all_total_rewards = np.array(all_total_rewards)
        mean_reward =  np.mean(all_total_rewards)
        max_reward = np.max(all_total_rewards)
        min_reward = np.min(all_total_rewards)
        std_reward = np.std(all_total_rewards)
        
        self.writer.add_scalars("reward",{"reward average":mean_reward,"reward max":max_reward,"reward min":min_reward,"reward std":std_reward},self.steps)
        self.writer.add_scalars("Q mean",{"Q mean average":Q_mean_mean,"Q mean max":Q_mean_max,"Q mean min":Q_mean_min,"Q mean std":Q_mean_std},self.steps)

        if(mean_reward > self.highest_mean_reward):
            self.highest_mean_reward = mean_reward
            torch.save(self.online_network.state_dict(),"model_avg"+'{:.2f}'.format(mean_reward) +"_min"+ '{:.2f}'.format(min_reward) + "_max"+ '{:.2f}'.format(max_reward) + "_std"+ '{:.2f}'.format(std_reward) + ".dat")

        for i in range(self.average):
            if(len(self.replay_memory) >= self.batch_size and self.steps >= self.steps_before_training):
                self.train_step()
            else:
                break

    def reset_env_done(self) -> None:
        """
        called when enviroment is in terminal state
        """

        self.test_agent()
        
        self.current_observation = self.env.reset()

        


    def step(self) -> None:
        """
        step in the enviroment by agent (followed by a training step)
        """
        #information collection
        state = self.current_observation
        self.online_network.resample_noise() # resample noise before taking action (when training) see noisy network paper
        action = self.choose_action(state)
        next_state,reward, done,_ = env.step(action)
        self.current_observation = next_state #update current observation
        self.replay_memory.append((state, next_state, action, reward, done)) #put information in memory

        if(done):
            self.reset_env_done()

        #train
        if(len(self.replay_memory) >= self.batch_size and self.steps >= self.steps_before_training):
            self.train_step()
        self.steps += 1
    
    
    def train_step(self)-> None:
        """
        Deep Reinforcement Learning with Double Q-learning 
        https://arxiv.org/abs/1509.06461
        the online network chooses the best action given the next state and the corresponding target
        is selected from the target network
        """
        optimizer.zero_grad()

        #unpack experience
        batch = self.replay_memory.sample(self.batch_size)
        states, next_states, actions, rewards, dones = batch

        #resample networks
        self.online_network.resample_noise()
        self.target_network.resample_noise()
        #estimate 
        estimated_Q = self.online_network(states).gather(1,actions.view(-1,1))
        

        with torch.no_grad():
            self.online_network.resample_noise() #for actions (see paper)
            next_actions = torch.argmax(self.online_network(next_states),dim=1)
            target_Q = self.target_network(next_states).gather(1,next_actions.view(-1,1))
            
        target_Q[dones] = 0.0 #terminal state has no future
         
        target = rewards.view(-1,1) + self.gamma * target_Q
       
        
        loss = nn.MSELoss()(estimated_Q,target)
        loss.backward()

        optimizer.step()
        self.writer.add_scalar("loss",loss.item(),self.steps)
       



    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def choose_action(self,obs:torch.FloatTensor,return_Q=False) -> tuple or int:
        Q_values = self.online_network(observation_converter(obs))
        
        if(return_Q):
            return (torch.argmax(Q_values).item(),Q_values.max().item())
        else:
            return torch.argmax(Q_values).item()

class mish(nn.Module):
    """
    activation function
    Mish: A Self Regularized Non-Monotonic Activation Function (https://arxiv.org/abs/1908.08681)
    
    """
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        return x * torch.tanh(F.softplus(x))

class noisy_layer(nn.Module):
    """
    Noisy layers (Factorised Gaussian noise)
    NOISY NETWORKS FOR EXPLORATION
    https://arxiv.org/abs/1706.10295

    NOTES:
    (training)
    resample target and resample online (they have different samples of noise for the weights)
    train the entire batch (with the current sample)
    (take action)
    resample online net then take an action
    """
    def __init__(self,input_size:int,output_size:int,sigma_zero=0.5):
        """
        sigma_zero (see paper under initialization (of variance weights) for Factorised parameters)
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.register_buffer("sigma_zero",torch.tensor(sigma_zero,dtype=torch.float) ) #register sigma zero (adds it to state dict of module)

        self.linear = nn.Linear(input_size,output_size)
        
        self.noisy_mean_weights = nn.Parameter(torch.Tensor(self.output_size, self.input_size))
        self.noisy_variance_weights = nn.Parameter(torch.Tensor(self.output_size, self.input_size)) 
        self.noisy_mean_bias = nn.Parameter(torch.Tensor(self.output_size)) 
        self.noisy_variance_bias = nn.Parameter(torch.Tensor(self.output_size))  

        self.noise_weights = None 
        self.noise_bias = None 

        self.reset_parameters() #initialize weights
        self.resample_noise() #initialization of noisy_weight and noisy_bias  
        

    def reset_parameters(self) -> None:
        """
        mean initialized with Uniform from [- 1/sqrt(p), 1/sqrt(p)]  where p is the input_size 
        variance initialized with a constant 
        """
        #mean
        init.uniform_(self.noisy_mean_weights, a=-(1/np.sqrt(self.input_size)), b=1/np.sqrt(self.input_size))
        init.uniform_(self.noisy_mean_bias, a=-(1/np.sqrt(self.input_size)), b=1/np.sqrt(self.input_size))
        #variance
        init.constant_(self.noisy_variance_weights,self.sigma_zero/np.sqrt(self.input_size))
        init.constant_(self.noisy_variance_bias,self.sigma_zero/np.sqrt(self.input_size))
        #linear
        self.linear.reset_parameters()
        

    def f(self,x:torch.FloatTensor) -> torch.FloatTensor:
        return torch.sign(x)*torch.sqrt(torch.abs(x))

    def resample_noise(self) -> None:
        """
        (Deep Q-Networks (DQN) and Dueling; section from the paper)
        
        The parameters are drawn from the noisy network parameter distribution after every replay step.  
        
        For replay, the current noisy network parameter sample is held fixed across the batch. Since DQN
        and Dueling take one step of optimisation for every action step, the noisy network parameters are
        re-sampled before every action.
        """

        self.noise_weights = torch.matmul(self.f(torch.cuda.FloatTensor(self.output_size,1).normal_()),self.f(torch.cuda.FloatTensor(1,self.input_size).normal_())) 
    
        self.noise_bias = self.f(torch.cuda.FloatTensor(self.output_size).normal_())
    
    
    def calc_noise_parameters(self) -> tuple:
        """
        calculate noisy parameters (without resamping the noise)
        returns: (noisy_weight, noisy_bias)
        """
       
        noisy_weight = self.noisy_mean_weights + self.noisy_variance_weights * self.noise_weights 
        
        noisy_bias = self.noisy_mean_bias + self.noisy_variance_bias * self.noise_bias
        return (noisy_weight,noisy_bias)

    def forward(self,x:torch.FloatTensor) -> torch.FloatTensor:
        
        return self.linear(x) + F.linear(x,*self.calc_noise_parameters())

if __name__ == "__main__":
    env = wrapper.wrap_deepmind(wrapper.make_atari("SpaceInvaders-v0",skip=3),frame_stack=True) #skip 4 in space invaders makes the lasers invisable

    net = network_conv() #4 input channels, 6 discrete actions

    optimizer = optim.Adam(net.parameters(),lr=0.0001) 
    agent = dqn_agent(net,env,optimizer,steps_before_training=0)
    sync_step = 1000 #sync every n steps

    for i in tqdm(range(10000000),ascii=True):
        agent.step()
        
        if(i % sync_step == 0):
            agent.sync_target_network()