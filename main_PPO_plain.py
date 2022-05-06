# EV Routing Charging

#C:\Users\yjxu\PycharmProjects\EV_traffic\venv\Lib\site-packages\gym\envs\algorithmic

import torch as th

import networkx as nx
import numpy as np
import pandas as pd
import gym
import tensorflow as tf
from ev_routine_test_env import ev_routine_env
from ev_routine_testplain_env import ev_routine_plain_env

# import EV1
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.vec_env import DummyVecEnv
import tf_slim as slim


import os
import numpy as np

# CPU的执行时间
start = time.time()
#
best_mean_reward, n_steps = -np.inf, 0
## Settings
ENVIRONMENT = 'ev-v1' # ev_routine_env
# 'ev-v0'

TIMEVALUE= 17

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
#print(np.shape(EV_velocity)) #(76, 2000)
#print(np.size(EV_velocity))
chargingprice = np.array(pd.read_csv('EV_price.csv'))
waitingtime = np.array(pd.read_csv('EV_waitingtime.csv'))
#print(np.shape(waitingtime)[1])

if ENVIRONMENT == 'ev-v0' :
    env = ev_routine_env(edgelist=edgelist, chargingstation=chargingstation,
                   edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime,beta_time=TIMEVALUE)
elif ENVIRONMENT == 'ev-v1' :
    env = ev_routine_plain_env(edgelist=edgelist, chargingstation=chargingstation,
                   edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime,
                               beta_time=TIMEVALUE)
    # gym.make(ENVIRONMENT,edgelist=edgelist, chargingstation=chargingstation,
    #            edgelength=edgelength, EV_velocity=EV_velocity,chargingprice=chargingprice,waitingtime=waitingtime)

#env = DummyVecEnv([lambda: env])

#policy_kwargs = dict(act_fun=tf.nn.tanh, net_arch=[128, 128, 128])
policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128, 128])

model = PPO("MlpPolicy", env, learning_rate=0.0003, verbose=1,tensorboard_log="./numerical_PPO/", policy_kwargs= policy_kwargs)
# tensorboard --logdir ./numerical_PPO/
# http://localhost:6006


# model = PPO("MlpPolicy", env, gamma=0.99,  #policy_kwargs=policy_kwargs,
#              verbose=1, tensorboard_log="./numerical_PPO/")

#model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./numerical_PPO/",policy_kwargs=policy_kwargs,)


#model.learn(total_timesteps=1000, tb_log_name="PPO"+ENVIRONMENT)
model.learn(total_timesteps=200000, tb_log_name="PPO"+str(TIMEVALUE)+ENVIRONMENT)
model.save(ENVIRONMENT+'PPOmodel'+str(TIMEVALUE))

episodeNum = 10000
se = 11111111111
total_reward = np.zeros(episodeNum)
optimal_reward= np.zeros(episodeNum)
oneshot= np.zeros(episodeNum)
rolling_cost = np.zeros(episodeNum)
for i in range(episodeNum):
    obs = env.reset()
    done = False
    total_reward[i] = 0.0
    oneshot[i] = env.oneshot()
    rolling_cost[i] = -env.rolling()
    optimal_reward[i] = -env.offline()
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info, = env.step(action)
        total_reward[i] += reward
    se += 1

end = time.time()
print ("运行时间为 :", end-start)
print(ENVIRONMENT+str(TIMEVALUE),'PPOtotal_reward is', np.sum(total_reward)/episodeNum)
print(ENVIRONMENT,'oneshot is', np.sum(oneshot)/episodeNum)
print(ENVIRONMENT,'rolling is', np.sum(rolling_cost)/episodeNum)

#print(ENVIRONMENT,'optimal_reward is', np.sum(optimal_reward)/episodeNum)








            # tensorboard_log="./numerical_PPO/")
# http://localhost:6006
# cd C:\Users\1000877\Documents\EV_charging\TSG_paper
# #C:\Users\1000877\Documents\EV_charging\exercises
# tensorboard --logdir ./numerical_EV1data91_tensorboard/
# tensorboard --logdir ./numerical_PPO/

#tensorboard --logdir ./TRPO_EV1data91_tensorboard/

# model.learn(total_timesteps=1000000,  callback=callback, tb_log_name="PPO"+ENVIRONMENT+str(N))


# model.learn(total_timesteps=1000000,  callback=callback, tb_log_name="PPO"+ENVIRONMENT+str(N))

#
# del model  # remove to demonstrate saving and loading
# del env
#
# env = gym.make(ENVIRONMENT,N=N)
# env = DummyVecEnv([lambda: env])
#
# model = PPO1.load(ENVIRONMENT+'model'+str(N))

#env = gym.make('EVdirectdata91-v0', data_ev=df_ev, data_solar_price=df_others, traintest=0, chargernumber=C_N, penetrationRate=C_p)





