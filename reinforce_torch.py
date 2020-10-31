
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class PolicyNetwork(nn.Module):
    def __init__(self, lr, input_dims, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return(x)

class PolicyGradientAgent:
    def __init__(self, lr, input_dims, gamma, n_actions):
        self.gamma = gamma
        self.lr = lr
        self.reward_memory = []
        self.action_memory = []
        self.policy = PolicyNetwork(self.lr, input_dims, n_actions)
        self.behavior = {}
        self.my_hp_memmory = 3
        self.enemy_hp_memmory = 3

    def choose_action(self, observation):
        state = T.Tensor([observation]).to(self.policy.device)
        prob = F.softmax(self.policy.forward(state), dim = 0)
        action_probs = T.distributions.Categorical(prob)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)

        return(action.item()) 

    def store_rewards(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()
        G = np.zeros_like(self.reward_memory, dtype=np.float64) 

        for i in reversed(range(len(self.reward_memory))):
            if self.reward_memory[i] != 0:
                reward = self.reward_memory[i]
            else:
                reward *= self.gamma
                self.reward_memory[i] = reward

        loss = 0

        for g, logprob in zip(self.reward_memory, self.action_memory):
            loss += -g * logprob
        print(loss)

        # for t in range(len(self.reward_memory)):
        #     G_sum = 0
        #     discount = 1
        #     for k in range(t, len(self.reward_memory)):
        #         G_sum += self.reward_memory[k] * discount
        #         discount *= self.gamma
        #     G[t] = G_sum
        # mean = np.mean(G)
        # std = np.std(G) if np.std(G) > 0 else 1
        # G = (G - mean)/std

        # G = T.tensor(G, dtype=T.float).to(self.policy.device)

        # loss = 0

        # for g, logprob in zip(G, self.action_memory):
        #     loss += -g * logprob
        # print(loss)
        
        loss.backward()
        self.policy.optimizer.step()

        self.action_memory = []
        self.reward_memory = []

    def get_observation(self, data: dict):
        self.observation = []
        self.enemies = data['enemies']
        if not self.enemies:
            self.enemies = [(0.0, 0.0)]
        self.hp = data['hp']
        self.r_vis = data['r_vis']
        self.rays_intersected = data['rays_intersected']
        self.rounds = data['rounds']
        if not self.rounds:
            self.rounds = [(0.0, 0.0)]
        self.theta = data['theta']
        self.time = data['time']
        
        self.observation.extend([self.hp, self.r_vis, self.theta, *self.rays_intersected])
        self.observation.extend(self.enemies[0])
        self.observation.extend(self.rounds[0])
        
        return np.array(self.observation)
    
    def get_reward(self, data: dict):
        my_hp = data['signals_comands'][-1][0]['hp']
        enemy_hp = data['signals_comands'][0][0]['hp']
        if my_hp < self.my_hp_memmory:
            self.my_hp_memmory = my_hp
            return -1
        elif enemy_hp < self.enemy_hp_memmory:
            self.enemy_hp_memmory = enemy_hp
            return 1
        else:
            return 0
        
    def from_numb_to_action(self, numb: int) -> dict:
        ''' Функция которая принимает на вход число в соотвествии с выходным слоем и переводит это в словарь с действием

        '''
        if numb == 0:
            self.behavior['move'] = 1
        elif numb == 1:
            self.behavior['move'] = -1
        elif numb == 2:
            self.behavior['rotate'] = -1
        elif numb == 3:
            self.behavior['rotate'] = 1
        elif numb == 4:
            self.behavior['fire'] = 1
        elif numb == 5:
            self.behavior['fire'] = 0
        elif numb == 6:
            self.behavior['vision'] = 1                    
        elif numb == 7:
            self.behavior['vision'] = -1

        return self.behavior.copy()


        
# {'move': 1, 'rotation': 0, 'fire': 0, 'vision': 0}
# {'vision': 1}
# {'move': 1, 'rotation': 0, 'fire': 0, 'vision': 1}


