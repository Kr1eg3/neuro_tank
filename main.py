from engine import Engine, random_behaivor
from myobject import MyObject
#from myobject2 import MyObject2
from myobject3 import MyObject3
from reinforce_torch import PolicyGradientAgent
import random
from pprint import pprint
import numpy as np
import torch as T


obj = MyObject(fullhp=3, nrays=10)
#obj2 = MyObject2(fullhp=3, nrays=10)
obj3 = MyObject3(fullhp=3, nrays=10)
agent = PolicyGradientAgent(gamma=0.99, lr=0.001, input_dims=[17], n_actions=8)

def ai_behavior(input_dict):
    global agent
    behavior = agent.get_observation()
    return behavior
    
def my_foo3(data: dict) -> dict:
    global obj3
    obj3.update_info(data)
    behaivor = obj3.get_behaivor()
    if not obj3.rollback:
        obj3.build_traj(behaivor)
    return behaivor

def my_foo(data: dict) -> dict:
    global obj
    obj.update_info(data)
    behaivor = obj.get_behaivor()
    if not obj.rollback:
        obj.build_traj(behaivor)
    return behaivor

def rotation(input_dict):
    target = input_dict.get('enemies')  # enemies - список в котором кортеж
    Steprotation = None
    if len(target) == 0:
        Steprotation = {'rotate': 1, 'fire': 0, 'vision': -1}
    else:
        moo = target[0]
        if moo[0] < 0:
            Steprotation = {'rotate': -0.5, 'fire': 1, 'vision': -0.8}
        elif moo[0] == 0:
            Steprotation = {'rotate': 0, 'fire': 1, 'vision': -0.8}
        elif moo[0] > 0:
            Steprotation = {'rotate': 0.5, 'fire': 1, 'vision': -0.8}
    return Steprotation

def create_brain(data: dict):
    def new_brain(x):
        return data
    return new_brain

def main():
    im = Engine.get_standart(you_brain_foo=my_foo3, enemy_brain_foo=my_foo3)
    var = 0 
    while not im.done:
        #print(ai_behavior())
        var += 1
        #if var == 200:
            #im.units[0].brain_foo = create_brain({'move': 1, 'rotate': -1, 'fire': 1, 'vision': -1})

        #im = im.get_standart(you_brain_foo=my_foo3, enemy_brain_foo=my_foo_test)
        info = im.step(render=True)
        info = info['signals_comands'][-1][0]
        #pprint(info)
        pprint(agent.get_observation(info))
        
    result = im.get_result()
    print(result)

'''if __name__ == '__main__':
    main()
else:
    print("Error: no entering point!")
    exit(-1)'''

'''def plot_learning_curve(scores, x, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)'''

if __name__ == '__main__':
    n_games = 3000
    scores = []
    
    for i in range(n_games):
        im = Engine.get_standart(you_brain_foo=my_foo3, enemy_brain_foo=my_foo3)
        info = im.step(render=True)
        info = info['signals_comands'][-1][0]
        observation = agent.get_observation(info)
        score = 0
        while not im.done:
            action = agent.from_numb_to_action(agent.choose_action(observation)) 
            im.units[1].brain_foo = create_brain(action)
            info = im.step(render=True)
            #pprint(info)
            info_ = info['signals_comands'][-1][0]
            observation = agent.get_observation(info_)
            score += agent.get_reward(info)
            # print(agent.choose_action(observation))
            # print(action)
            agent.store_rewards(agent.get_reward(info))
        agent.reward_memory[-1] = im.get_result()
        # print(len(agent.reward_memory))
        # print(len(agent.action_memory))
        agent.learn()
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        print('episode ', i, 'score %.2f' % score,
                'average score %.2f' % avg_score)

'''    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(scores, x, figure_file)'''