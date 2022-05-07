# main_A2C
# EV Routing Charging, SP+A2C


#C:\Users\yjxu\PycharmProjects\EV_traffic\venv\Lib\site-packages\gym\envs\algorithmic


import networkx as nx
import numpy as np
import pandas as pd
import gym
from ev_routine_test_env import ev_routine_env
from ev_routine_testplain_env import ev_routine_plain_env
import torch as th
import torch.nn as nn
import tensorflow as tf
# import EV1
import time
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv
import tf_slim as slim


import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# CPU的执行时间
start = time.time()
#
best_mean_reward, n_steps = -np.inf, 0
## Settings
ENVIRONMENT = 'ev-v0'  #'ev-v0'
TIMEVALUE= 11

# 'evoptimal-v0',  'ev-v0'
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
#env = DummyVecEnv([lambda: env])

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

policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128,128,128])

model = A2C("MlpPolicy", env, learning_rate=0.0003, verbose=1, tensorboard_log="./numerical_A2C/", policy_kwargs= policy_kwargs)
# tensorboard --logdir ./numerical_A2C/
# http://localhost:6006

model.learn(total_timesteps=2000000, tb_log_name="A2C"+ENVIRONMENT, callback=TensorboardCallback())
model.save(ENVIRONMENT+'A2Cmodel0'+str(TIMEVALUE))

end = time.time()

episodeNum = 1000
se = 11111111111
total_reward = np.zeros(episodeNum)
for i in range(episodeNum):
    obs = env.reset()
    done = False
    total_reward[i] = 0.0
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info, = env.step(action)
        total_reward[i] += reward
    se += 1


print ("运行时间为 :", end-start)
print(ENVIRONMENT+str(TIMEVALUE),'A2Ctotal_reward is', np.sum(total_reward)/episodeNum)








