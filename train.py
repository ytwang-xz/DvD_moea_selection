
from policies import FullyConnected
from learner import Learner
from utils import *
from es import population_update
from experiments import get_experiment
import simpleenvs
from bandits import BayesianBandits
from embeddings import embed

import psutil
import ray
import gym
import parser
import argparse
import numpy as np
import pandas as pd

import os
import time
from copy import deepcopy
from random import sample 
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from dppy.finite_dpps import FiniteDPP


def select_states(master, params, states):

    # select random states... if you dont have enough just take the ones you have
    if int(params['states'].split('-')[1]) < len(states):
        selected = sample(states, int(params['states'].split('-')[1]))
        return(selected)
    else:
        return(states)

# needed because sometimes the plasma store fills up... ray issue don't FULLY understand :)
def reset_ray(master, params):
    ray.disconnect()
    ray.shutdown()
    time.sleep(5)
    del os.environ['RAY_USE_NEW_GCS']
    ray.init(
        plasma_directory="/tmp")
    os.environ['RAY_USE_NEW_GCS'] = 'True'
    flush_policy = ray.experimental.SimpleGcsFlushPolicy(flush_period_secs=0.1)        
    ray.experimental.set_flushing_policy(flush_policy)
                
def train(params):
    if params['moea_select'] == 1:
        env = gym.make(params['env_name'])
        print(hasattr(env, 'tasks'))
        params['ob_dim'] = env.observation_space.shape[0]
        params['ac_dim'] = env.action_space.shape[0]

        master = Learner(params)
        # 实例化的时候，已经得到了初始策略
        # self.agents = {i:get_policy(params, params['seed']+1000*i) for i in range(params['num_agents'])}
        master.reward = {i: -9999 for i in range(params['num_agents'])}

        n_eps = 0
        n_iter = 0
        ts_cumulative = 0
        ts, rollouts, rewards, max_rwds, dists, min_dists, agents, lambdas = [0], [0], [], [], [], [], [], []
        params['num_sensings'] = params['sensings']
        # 这个参数表示ES里面采样的个数: 100*2

        master.agent = 0
        # get initial states so you can get behavioral embeddings
        # Agents以字典形式储存， 每一个Agent是一个全连接网络
        # dict_keys([0, 1, 2, 3, 4])

        # population = [master.agents[x].rollout(env, params['steps'], incl_data=True) for x in master.agents.keys()]
        # population 的每个元素是一个tuple，tuple的第一个元素是reward, 第二个元素是[state, reward, action]

        # 每个个体评估num_evals次
        population = []
        if params['num_evals'] > 0:
            seeds = [int(np.random.uniform() * 10000) for _ in range(params['num_evals'])]

        for i in master.agents.keys():
            if params['num_evals'] > 0:
                reward = 0
                for j in range(params['num_evals']):
                    r, traj = master.agents[i].rollout(env, params['steps'], incl_data=True, seed=seeds[j])
                    # 环境可以设置不同seed
                    reward += r
                reward /= params['num_evals']
            else:
                reward, traj = master.agents[i].rollout(env, params['steps'], incl_data=True)
            master.reward[i] = reward
            population.append(traj)
        # population是长度为5的list, 每个元素为array, 每个array里面包含了1000个样本[state, reward, action]
        all_states = [s[0] for x in population for s in x]
        # all_states表示所有的Agent访问过的所有状态，可能有重复
        master.selected = select_states(master, params, all_states)
        # 随机选20个
        master.update_embeddings(params)
        # master.embeddings[j]表示第j个策略在选定策略上的动作，array
        master.calc_pairwise_dists(params)
        # 计算各个策略之间的l_2距离， 保存在self.dists， self.min_dist， self.dist_vec中
        # master.select_agent()
        master.agent = np.argmax(np.random.multinomial(1, master.dist_vec))
        # 通过采样的方法选出最diverse的Agent, 标号保存在self.agent中
        master.get_agent()
        # self.policy设为前面找到的agent
        # initial reward
        reward = master.policy.rollout(env, params['steps'], incl_data=False)

        rewards.append(reward)
        agents.append(master.agent)
        dists.append(master.dists)
        max_reward = reward
        max_rwds.append(max_reward)
        min_dists.append(master.min_dist)

        if params['w_nov'] < 0:
            bb = BayesianBandits()
            params['w_nov'] = 0
        lambdas.append(params['w_nov'])

        while n_iter < params['max_iter']:

            print('Iter: %s, Eps: %s, Mean: %s, Max: %s, Best: %s, MeanD: %s, MinD: %s, Lam: %s' % (
            n_iter, n_eps, np.round(reward, 4), np.round(max_reward, 4), master.agent, np.round(master.dists, 4),
            np.round(master.min_dist, 4), params['w_nov']))

            if (n_iter > 0) & (params['num_agents'] > 1):
                master.calc_pairwise_dists(params)
            #     master.select_agent()
            #     master.get_agent()

            ## Main Function Call
            params['n_iter'] = n_iter

            if params['num_agents'] > 1:
                gradient, timesteps = population_update(master, params)
                # self.buffer = [] // 更新第0轮个体时所有子代个体的轨迹样本
                # self.states = [] // 更新第0轮个体时所有子代个体的状态

                # timesteps = 100 0000
                n_eps += 2 * params['num_sensings'] * params['num_agents']

                # n_eps = 1000
            else:
                gradient, timesteps = individual_update(master, params)
                n_eps += 2 * params['num_sensings']

            ts_cumulative += timesteps
            all_states += master.states
            if params['num_sensings'] < len(all_states):
                all_states = sample(all_states, params['num_sensings'])

            gradient /= (np.linalg.norm(gradient) / master.policy.N + 1e-8)

            n_iter += 1
            update = Adam(gradient, master, params['learning_rate'], n_iter)
            # Adam是如何更新的：
            # learner.m = beta1 * learner.m + (1 - beta1) * dx
            # mt = learner.m / (1 - beta1 ** t)
            # learner.v = beta2 * learner.v + (1 - beta2) * (dx ** 2)
            # vt = learner.v / (1 - beta2 ** t)
            # update = learning_rate * mt / (np.sqrt(vt) + eps)
            # return (update)

            rwds, trajectories = [], []
            if params['num_evals'] > 0:
                seeds = [int(np.random.uniform() * 10000) for _ in range(params['num_evals'])]
            master.init_population(params)
            # 将父代个体copy到population中
            print("更新种群完成")
            for i in range(params['num_agents']):
                master.agent = i
                master.get_agent()
                # self.policy = deepcopy(self.agents[self.agent])
                # self.embedding = self.embeddings[self.agent].copy()
                # self.m = self.adam_params[self.agent][0]
                # self.v = self.adam_params[self.agent][1]
                master.policy.update(master.policy.params + update[(i * master.policy.N):((i + 1) * master.policy.N)])
                if params['num_evals'] > 0:
                    reward = 0
                    for j in range(params['num_evals']):
                        r, traj = master.policy.rollout(env, params['steps'], incl_data=True, seed=seeds[j])
                        # 环境可以设置不同seed
                        reward += r
                    reward /= params['num_evals']
                else:
                    reward, traj = master.policy.rollout(env, params['steps'], incl_data=True)
                # rwds.append(reward)
                master.reward_population[i+params['num_agents']] = reward
                trajectories.append(traj)
                # if reward > master.best[i]:
                #     master.best[i] = reward
                #     np.save('data/%s/weights/Seed%s_Agent%s' % (params['dir'], params['seed'], i), master.policy.params)
                # master.reward[i].append(reward)
                master.embedding = embed(params, traj, master.policy, master.selected)
                master.update_child(params)
                print("更新子代完成")
            master.get_dist(params)
            print("计算两两之间距离完成")
            print("population reward:")
            print(master.reward_population)
            print("population distance:")
            print(master.dist_population)
            print("Adam parameters:")
            print(master.adam_params_population)
            master.update_agents(params)
            for i in range(params['num_agents']):
                rwds.append(master.reward[i])
            reward = np.mean(rwds)
            max_reward = max(rwds)
            traj = trajectories[np.argmax(rwds)]
            master.agent = np.argmax(rwds)

            # Update selected states
            master.selected = select_states(master, params, all_states)
            master.update_embeddings(params)

            # master.embedding = embed(params, traj, master.policy, master.selected)
            rewards.append(reward)
            max_rwds.append(max_reward)
            # master.reward[master.agent].append(reward)
            # if reward > master.best[master.agent]:
            #     master.best[master.agent] = reward
            #     np.save('data/%s/weights/Seed%s_Agent%s' % (params['dir'], params['seed'], master.agent),
            #             master.policy.params)

            ## update the bandits
            try:
                bb.update_dists(reward)
                params['w_nov'] = bb.sample()
                # print(params['w_nov'])
            except NameError:
                pass

            lambdas.append(params['w_nov'])
            rollouts.append(n_eps)
            agents.append(master.agent)
            dists.append(master.dists)
            min_dists.append(master.min_dist)
            ts.append(ts_cumulative)
            # master.update_agent()

            if n_iter % params['flush'] == 0:
                reset_ray(master, params)
                master.init_workers(params)

            out = pd.DataFrame(
                {'Rollouts': rollouts, 'Reward': rewards, 'Max': max_rwds, 'Timesteps': ts, 'Dists': dists,
                 'Min_Dist': min_dists, 'Agent': agents, 'Lambda': lambdas})
            out.to_csv('data/%s/results/Seed%s.csv' % (params['dir'], params['seed']), index=False)

    else:
        env = gym.make(params['env_name'])
        params['ob_dim'] = env.observation_space.shape[0]
        params['ac_dim'] = env.action_space.shape[0]

        master = Learner(params)
        master.reward = {i: [-9999] for i in range(params['num_agents'])}

        n_eps = 0
        n_iter = 0
        ts_cumulative = 0
        ts, rollouts, rewards, max_rwds, dists, min_dists, agents, lambdas = [0], [0], [], [], [], [], [], []
        params['num_sensings'] = params['sensings']

        master.agent = 0
        # get initial states so you can get behavioral embeddings
        # Agents以字典形式储存， 每一个Agent是一个全连接网络
        # dict_keys([0, 1, 2, 3, 4])
        population = [master.agents[x].rollout(env, params['steps'], incl_data=True) for x in master.agents.keys()]
        # population 的每个元素是一个tuple，tuple的第一个元素是reward, 第二个元素是[state, reward, action]
        all_states = [s[0] for x in population for s in x[1]]
        # all_states表示所有的Agent访问过的所有状态，可能有重复
        master.selected = select_states(master, params, all_states)
        master.update_embeddings(params, population)
        # master.embeddings[j]表示第j个策略在选定策略上的动作，array
        master.calc_pairwise_dists(params)
        # 计算各个策略之间的l_2距离， 保存在self.dists， self.min_dist， self.dist_vec中
        master.select_agent()
        # 通过采样的方法选出最diverse的Agent, 标号保存在self.agent中
        master.get_agent()
        # self.policy设为前面找到的agent
        # initial reward
        reward = master.policy.rollout(env, params['steps'], incl_data=False)

        rewards.append(reward)
        agents.append(master.agent)
        dists.append(master.dists)
        max_reward = reward
        max_rwds.append(max_reward)
        min_dists.append(master.min_dist)

        if params['w_nov'] < 0:
            bb = BayesianBandits()
            params['w_nov'] = 0
        lambdas.append(params['w_nov'])

        while n_iter < params['max_iter']:

            print('Iter: %s, Eps: %s, Mean: %s, Max: %s, Best: %s, MeanD: %s, MinD: %s, Lam: %s' % (
            n_iter, n_eps, np.round(reward, 4), np.round(max_reward, 4), master.agent, np.round(master.dists, 4),
            np.round(master.min_dist, 4), params['w_nov']))

            if (n_iter > 0) & (params['num_agents'] > 1):
                master.calc_pairwise_dists(params)
                master.select_agent()
                master.get_agent()

            ## Main Function Call
            params['n_iter'] = n_iter

            if params['num_agents'] > 1:
                gradient, timesteps = population_update(master, params)

                # timesteps = 100 0000
                n_eps += 2 * params['num_sensings'] * params['num_agents']

                # n_eps = 1000
            else:
                gradient, timesteps = individual_update(master, params)
                n_eps += 2 * params['num_sensings']

            ts_cumulative += timesteps
            all_states += master.states
            if params['num_sensings'] < len(all_states):
                all_states = sample(all_states, params['num_sensings'])

            gradient /= (np.linalg.norm(gradient) / master.policy.N + 1e-8)

            n_iter += 1
            update = Adam(gradient, master, params['learning_rate'], n_iter)

            rwds, trajectories = [], []
            if params['num_evals'] > 0:
                seeds = [int(np.random.uniform() * 10000) for _ in range(params['num_evals'])]
            for i in range(params['num_agents']):
                master.agent = i
                master.get_agent()
                # self.policy = deepcopy(self.agents[self.agent])
                # self.embedding = self.embeddings[self.agent].copy()
                # self.m = self.adam_params[self.agent][0]
                # self.v = self.adam_params[self.agent][1]
                master.policy.update(master.policy.params + update[(i * master.policy.N):((i + 1) * master.policy.N)])
                if params['num_evals'] > 0:
                    reward = 0
                    for j in range(params['num_evals']):
                        r, traj = master.policy.rollout(env, params['steps'], incl_data=True, seed=seeds[j])
                        reward += r
                    reward /= params['num_evals']
                else:
                    reward, traj = master.policy.rollout(env, params['steps'], incl_data=True)
                rwds.append(reward)
                trajectories.append(traj)
                if reward > master.best[i]:
                    master.best[i] = reward
                    np.save('data/%s/weights/Seed%s_Agent%s' % (params['dir'], params['seed'], i), master.policy.params)
                master.reward[i].append(reward)
                master.update_agent()
                # 如何更新的：
                # self.agents[self.agent] = deepcopy(self.policy)
                # self.embeddings[self.agent] = self.embedding.copy()
                # self.adam_params[self.agent] = [self.m, self.v]

            reward = np.mean(rwds)
            max_reward = max(rwds)
            traj = trajectories[np.argmax(rwds)]
            master.agent = np.argmax(rwds)

            # Update selected states
            master.selected = select_states(master, params, all_states)
            master.update_embeddings(params)

            master.embedding = embed(params, traj, master.policy, master.selected)
            rewards.append(reward)
            max_rwds.append(max_reward)
            master.reward[master.agent].append(reward)
            if reward > master.best[master.agent]:
                master.best[master.agent] = reward
                np.save('data/%s/weights/Seed%s_Agent%s' % (params['dir'], params['seed'], master.agent),
                        master.policy.params)

            ## update the bandits
            try:
                bb.update_dists(reward)
                params['w_nov'] = bb.sample()
            except NameError:
                pass

            lambdas.append(params['w_nov'])
            rollouts.append(n_eps)
            agents.append(master.agent)
            dists.append(master.dists)
            min_dists.append(master.min_dist)
            ts.append(ts_cumulative)
            master.update_agent()

            if n_iter % params['flush'] == 0:
                reset_ray(master, params)
                master.init_workers(params)

            out = pd.DataFrame(
                {'Rollouts': rollouts, 'Reward': rewards, 'Max': max_rwds, 'Timesteps': ts, 'Dists': dists,
                 'Min_Dist': min_dists, 'Agent': agents, 'Lambda': lambdas})
            out.to_csv('data/%s/results/Seed%s.csv' % (params['dir'], params['seed']), index=False)


def main():
    
    parser = argparse.ArgumentParser()

    ## Env setup
    parser.add_argument('--env_name', type=str, default='point-v0')
    parser.add_argument('--num_agents', '-na', type=int, default=5)
    parser.add_argument('--seed', '-sd', type=int, default=0)
    parser.add_argument('--max_iter', '-it', type=int, default=100)
    parser.add_argument('--policy', '-po', type=str, default='FC')
    parser.add_argument('--embedding', '-em', type=str, default='a_s')
    parser.add_argument('--num_workers', '-nw', type=int, default=4)
    parser.add_argument('--filename', '-f', type=str, default='')
    parser.add_argument('--num_evals', '-ne', type=int, default=0)
    parser.add_argument('--flush', '-fl', type=int, default=1000) # may need this. it resets ray, because sometimes it fills the memory.
    parser.add_argument('--ob_filter', '-ob', type=str, default='MeanStdFilter') # 'NoFilter'
    parser.add_argument('--w_nov', '-wn', type=float, default=-1) # if negative it uses the adaptive method, else it will be fixed at the value you pick (0,1). Note that if you pick 1 itll be unsupervised (ie no reward)
    parser.add_argument('--dpp_kernel', '-ke', type=str, default='rbf')
    parser.add_argument('--states', '-ss', type=str, default='random-20') # 'random-X' X is how many
    parser.add_argument('--update_states', '-us', type=int, default=20) # how often to update.. we only used 20
    parser.add_argument('--moea_select', '-mo', type=int, default=0)  # if use the moea_selection

    args = parser.parse_args()
    params = vars(args)

    params = get_experiment(params)

    ray.init()
    os.environ['RAY_USE_NEW_GCS'] = 'True'

    state_word = [str(params['states'].split('-')[0]) if params['w_nov'] > 0 else ''][0]  #state_world = ''
    params['dir'] = params['env_name'] + '_Net' + str(params['layers']) + 'x' + str(params['h_dim']) + '_Agents' + str(params['num_agents']) + '_Novelty' + str(params['w_nov']) + state_word + 'kernel_' + params['dpp_kernel'] + '_lr' + str(params['learning_rate']) + '_' + params['filename'] + params['ob_filter'] + '_moea_select' + '_' + str(params['moea_select'])
    if not(os.path.exists('data/'+params['dir'])):
        os.makedirs('data/'+params['dir'])
        os.makedirs('data/'+params['dir']+'/weights')
        os.makedirs('data/'+params['dir']+'/results')
    
    train(params)

if __name__ == '__main__':
    main()
