# main_DQN

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import sys
# sys.path.append('./')

#C:\Users\yjxu\PycharmProjects\EV_traffic\venv\Lib\site-packages\gym\envs\algorithmic

import torch as th

import networkx as nx
import numpy as np
import pandas as pd
import gym
from ev_routine_test_env import ev_routine_env
from ev_routine_testplain_env import ev_routine_plain_env

import tensorflow as tf
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.vec_env import DummyVecEnv
import tf_slim as slim


import os
import numpy as np

# CPU的执行时间
start = time.time()
#
best_mean_reward, n_steps = -np.inf, 0
## Settings
ENVIRONMENT = 'ev-v0'
TIMEVALUE=11


edgelist = [[1,2],[1,3],[2,1],[2,6],[3,1],[3,4],[3,12],[4,3],[4,5],[4,11],
            [5,4],[5,6],[5,9],[6,2],[6,5],[6,8],[7,8],[7,18],[8,6],[8,7],
            [8,9],[8,16],[9,5],[9,8],[9,10],[10,9],[10,11],[10,15],[10,16],[10,17],
            [11,4],[11,10],[11,12],[11,14],[12,3],[12,11],[12,13],[13,12],[13,24],[14,11],
            [14,15],[14,23],[15,10],[15,14],[15,19],[15,22],[16,8],[16,10],[16,17],[16,18],
            [17,10],[17,16],[17,19],[18,7],[18,16],[18,20],[19,15],[19,17],[19,20],[20,18],
            [20,19],[20,21],[20,22],[21,20],[21,22],[21,24],[22,15],[22,20],[22,21],[22,23],
            [23,14],[23,22],[23,24],[24,13],[24,21],[24,23]]

#print(np.size(edgelist)) #152
#print(np.shape(edgelist)) #(76,2)
edgelength_original = np.array(pd.read_csv("edge_length.csv"))
#print(np.size(edgelength)) #152
edgelength = edgelength_original[:,1]
#print(edgelength)

chargingstation = [4,7,20]
EV_velocity = np.array(pd.read_csv("EV_velocity_data.csv"))
#print(np.shape(EV_velocity)) #(76, 100)
#print(np.size(EV_velocity))
chargingprice = np.array(pd.read_csv('EV_price.csv'))
waitingtime = np.array(pd.read_csv('EV_waitingtime.csv'))
#print(np.shape(waitingtime)[1])


if ENVIRONMENT == 'ev-v0':
    env = ev_routine_env(edgelist=edgelist, chargingstation=chargingstation,
                   edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime,beta_time=TIMEVALUE)
elif ENVIRONMENT == 'ev-v1':
    env = ev_routine_plain_env(edgelist=edgelist, chargingstation=chargingstation,
                   edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime,
                               beta_time=TIMEVALUE)

# env = ev_routine_env(edgelist=edgelist, chargingstation=chargingstation,
#                edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime)
# env = gym.make(ENVIRONMENT,edgelist=edgelist, chargingstation=chargingstation,
#                edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime)
# env = DummyVecEnv([lambda: env])
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True
#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128, 128])
#policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, 128])
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128, 128])

model = DQN("MlpPolicy", env, verbose=1, gamma=0.99, learning_rate=0.0003,buffer_size=200000,
            batch_size=8,exploration_fraction=1,train_freq=10,
            exploration_final_eps=0.02, target_update_interval=500,max_grad_norm=10,
            tensorboard_log="./numerical_DQN/",policy_kwargs= policy_kwargs)
# tensorboard --logdir ./numerical_DQN/
# http://localhost:6006



# model = PPO("MlpPolicy", env, gamma=0.99,  #policy_kwargs=policy_kwargs,
#              verbose=1, tensorboard_log="./numerical_PPO/")

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./numerical_PPO/",policy_kwargs=policy_kwargs,)


#model.learn(total_timesteps=1000, tb_log_name="PPO"+ENVIRONMENT)
model.learn(total_timesteps=200000, tb_log_name="DQN"+str(TIMEVALUE)+ENVIRONMENT, callback=TensorboardCallback())
model.save(ENVIRONMENT+'DQNmodel'+str(TIMEVALUE))

episodeNum = 1000
se = 11111111111
total_reward = np.zeros(episodeNum)
for i in range(episodeNum):
    obs = env.reset()
    done = False
    total_reward[i] = 0.0
    #    offline_optimality2[i] = -env2.offline() # same with offline_optimality2
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info, = env.step(action)
        total_reward[i] += reward
    se += 1

end = time.time()
print ("运行时间为 :", end-start)
print(ENVIRONMENT,str(TIMEVALUE),'DQNtotal_reward is', np.sum(total_reward)/episodeNum)

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

