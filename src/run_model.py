import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
import gym 
import wrapper
import numpy as np 
from time import sleep
from gym import wrappers
import numpy as np 
from tqdm import tqdm

  
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

def choose_action(obs:torch.FloatTensor,network:network_conv) -> int:
    return torch.argmax(network(obs.cuda())).item()

if __name__ == "__main__":
    env = wrapper.wrap_deepmind(wrapper.make_atari("SpaceInvaders-v0",skip=3),frame_stack=True, scale=False,episode_life=False)

    
    net = network_conv()
    """
    model_avg47.00_min0.00_max97.00_std32.99.dat
    mean:  56.61
    max:  103.0
    min:  24.0
    std:  24.74263324708993
    """

    net.load_state_dict(torch.load("model_avg47.00_min0.00_max97.00_std32.99.dat"))
    net.cuda()
    

    score_list = []


    for i in tqdm(range(100),ascii=True):
        action = choose_action(observation_converter(env.reset()),network=net)

        total_reward = 0
        while(True):
            env.render()
            sleep(0.1)
            net.resample_noise()
            obs,reward, done,_ = env.step(action)
            total_reward += reward
            action = choose_action(observation_converter(obs),network=net)
            if(done):
                break
        
 
        score_list.append(total_reward)

    score_list = np.array(score_list)
    print("mean: ", np.mean(score_list))
    print("max: ",np.max(score_list) )
    print("min: ",np.min(score_list) )
    print("std: ",np.std(score_list) )
    