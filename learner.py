
## Requirements
from utils import *
from policies import get_policy
import simpleenvs
from worker import Worker
from embeddings import embed
from experiments import get_experiment
import argparse
## External
import ray
import gym
#import mujoco_py
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import rankdata
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from dppy.finite_dpps import FiniteDPP
import operator

class Learner(object):

    def __init__(self, params):

        params['zeros'] = False
        self.agents = {i:get_policy(params, params['seed']+1000*i) for i in range(params['num_agents'])}

        self.timesteps = 0

        self.w_reward = 1
        self.w_size = 0
        self.dists = 0

        self.adam_params = {i:[0,0] for i in range(params['num_agents'])}

        self.buffer = []
        self.states = []
        self.embeddings = {i:[] for i in range(params['num_agents'])}
        self.best = {i:-9999 for i in range(params['num_agents'])}

        self.min_dist = 0

        self.num_workers = params['num_workers']
        self.init_workers(params)

        self.population = {i:0 for i in range(2*params['num_agents'])}
        self.adam_params_population = {i: [0, 0] for i in range(2*params['num_agents'])}
        self.embeddings_population = {i: [] for i in range(2*params['num_agents'])}
        self.reward_population = {i:0 for i in range(2*params['num_agents'])}
        self.dist_population = {i:0 for i in range(2*params['num_agents'])}

    def init_workers(self, params):
        
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = params['seed'] + 3)
        
        self.workers = [Worker.remote(params['seed'] + 7 * i,
                                      env_name=params['env_name'],
                                      policy=params['policy'],
                                      h_dim = params['h_dim'],
                                      layers = params['layers'],
                                      deltas= deltas_id,
                                      rollout_length=params['steps'],
                                      delta_std=params['sigma'],
                                      num_evals=params['num_evals'],
                                      ob_filter=params['ob_filter']) for i in range(params['num_workers'])]
        # workers是一个列表，列表中的每个元素是一个class Worker


    def get_agent(self):
        self.policy = deepcopy(self.agents[self.agent])
        self.embedding = self.embeddings[self.agent].copy()
        self.m = self.adam_params[self.agent][0]
        self.v = self.adam_params[self.agent][1]

    def update_agent(self):
        self.agents[self.agent] = deepcopy(self.policy)
        self.embeddings[self.agent] = self.embedding.copy()
        self.adam_params[self.agent] = [self.m, self.v]
    
    def update_embeddings(self, params, data=[]):
        
        for j in range(params['num_agents']):
            if params['embedding'] == 'a_s':
                self.embeddings[j] = [embed(params, [], self.agents[j], self.selected)]
            else:
                self.embeddings[j] = [embed(params, s, self.agents[j], self.selected) for s in data[j][1]]

    def calc_pairwise_dists(self, params):
        
        dists = np.zeros([params['num_agents'], params['num_agents']])
        
        min_dist = 999
        for i in range(params['num_agents']):
            for j in range(params['num_agents']):
                dists[i][j] = np.linalg.norm(self.embeddings[i][0] - self.embeddings[j][0])
                if (i != j) & (dists[i][j] < min_dist):
                    min_dist = dists[i][j] 
    
        self.dists = np.mean(dists)
        # print(self.dists)
        self.min_dist = min_dist
        #对每一行算均值
        self.dist_vec = np.mean(dists, axis=1)
        #标准化
        self.dist_vec /= np.sum(self.dist_vec)

        
    def select_agent(self):

        if min([x[-1] for x in list(self.reward.values())]) > -9999:
            reward_vec = rankdata([max(x[-5:]) for x in list(self.reward.values())])
            reward_vec /= np.sum(reward_vec)
            
            dist_vec = rankdata(self.dist_vec)
            dist_vec /= np.sum(dist_vec)
            vec = (dist_vec + reward_vec)/2
            self.agent = np.argmax(np.random.multinomial(1, vec))
        else:
            self.agent = np.argmax(np.random.multinomial(1, self.dist_vec))

    #判断个体i是否支配j
    def dominates(self, i, j):
        if self.reward_population[i] > self.reward_population[j] and self.dist_population[i] >= self.dist_population[j]:
            return True
        elif self.reward_population[i] >= self.reward_population[j] and self.dist_population[i] > self.dist_population[j]:
            return True
        else:
            return False

    def fast_nondominated_sort(self, params):
        domination_count = [0 for i in range(2*params['num_agents'])]
        # 占优该个体的个数
        dominated_solutions = [[] for i in range(2*params['num_agents'])]
        # 被该个体占优的个体
        population_fronts = []
        # rank为0的个
        population_fronts.append([])
        rank = [100 for i in range(2*params['num_agents'])]

        for i in range(2*params['num_agents']):
            for j in range(2*params['num_agents']):
                if self.dominates(i,j):
                    dominated_solutions[i].append(j)
                elif self.dominates(j,i):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                population_fronts[0].append(i)
                rank[i] = 0
        # print(domination_count)
        # print(dominated_solutions)
        # print(population_fronts)
        # print(rank)
        k = 0
        while len(population_fronts[k]) > 0:
            temp = []
            for i in population_fronts[k]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        rank[j] = k + 1
                        temp.append(j)
            k = k + 1
            population_fronts.append(temp)
        return population_fronts

    def calculate_crowding_distance(self, front):
        l = len(front)
        crowding_distance = {}
        for i in front:
            crowding_distance[i] = 0
        if l <= 2:
            return sorted(crowding_distance.items(), key=operator.itemgetter(1), reverse=True)

        # 计算reward对应的crowding distance
        front.sort(key = self.reward_sort)
        crowding_distance[front[0]] = np.inf
        crowding_distance[front[l-1]] = np.inf
        normalization = self.reward_population[front[l-1]] - self.reward_population[front[0]]
        for i in range(1,l-1):
            if normalization == 0:
                continue
            crowding_distance[front[i]] = crowding_distance[front[i]] + abs((self.reward_population[front[i+1]] - self.reward_population[front[i-1]])/normalization)

        # 计算dist对应的crowding distance
        front.sort(key=self.dist_sort)
        crowding_distance[front[0]] = np.inf
        crowding_distance[front[l - 1]] = np.inf
        normalization = self.dist_population[front[l - 1]] - self.dist_population[front[0]]
        for i in range(1, l - 1):
            if normalization == 0:
                continue
            crowding_distance[front[i]] = crowding_distance[front[i]] + abs(
                (self.dist_population[front[i + 1]] - self.dist_population[front[i - 1]]) / normalization)
        # crowding_distance = sorted(crowding_distance.items(), key=lambda x: x[1], reverse=True)
        crowding_distance = sorted(crowding_distance.items(), key=operator.itemgetter(1), reverse=True)
        #
        # sorted(crowding_distance, key=lambda kv: (kv[1], kv[0]))
        return crowding_distance

    def reward_sort(self, elem):
        return self.reward_population[elem]

    def dist_sort(self, elem):
        return self.dist_population[elem]

    def moea_selection(self, params):
        fronts = self.fast_nondominated_sort(params)
        new_population = []
        front_num = 0
        while len(new_population) + len(fronts[front_num]) <= params['num_agents']:
            if len(fronts[front_num]) > 0:
                for i in fronts[front_num]:
                    new_population.append(i)

                if len(new_population) == params['num_agents']:
                    continue
            front_num += 1
        if len(new_population) < params['num_agents']:
            crowding_distance = self.calculate_crowding_distance(fronts[front_num])
            print(crowding_distance)
            # 返回的是列表，因为字典没有顺序 [(1, inf), (9, inf), (5, 1.341557202181469), (7, 0.8228518836836001)]
            for item in crowding_distance:
                new_population.append(item[0])
                if len(new_population) == params['num_agents']:
                    break
        return new_population

    def update_agents(self, params):
        new_population_idx = self.moea_selection(params)
        print("选中的个体：")
        print(new_population_idx)
        for i in range(params['num_agents']):
            self.agents[i] = deepcopy(self.population[new_population_idx[i]])
            self.adam_params[i] = self.adam_params_population[new_population_idx[i]]
            self.embeddings[i] = self.embeddings_population[new_population_idx[i]]
            self.reward[i] = self.reward_population[new_population_idx[i]]
            self.dist_vec[i] = self.dist_population[new_population_idx[i]]

    def init_population(self, params):
        for i in range(params['num_agents']):
            self.population[i] = deepcopy(self.agents[i])
            self.adam_params_population[i] = self.adam_params[i]
            self.embeddings_population[i] = self.embeddings[i]
            self.reward_population[i] = self.reward[i]
            self.dist_population[i] = self.dist_vec[i]

    def update_child(self, params):
        self.population[self.agent + params['num_agents']] = deepcopy(self.policy)
        self.embeddings_population[self.agent + params['num_agents']] = self.embedding.copy()
        self.adam_params_population[self.agent + params['num_agents']] = [self.m, self.v]

    def get_dist(self, params):
        dists = np.zeros([2*params['num_agents'], 2*params['num_agents']])

        min_dist = 999
        for i in range(2*params['num_agents']):
            for j in range(2*params['num_agents']):
                dists[i][j] = np.linalg.norm(self.embeddings_population[i][0] - self.embeddings_population[j][0])
                if (i != j) & (dists[i][j] < min_dist):
                    min_dist = dists[i][j]

        p = np.mean(dists, axis=1)
        # 标准化
        p /= np.sum(p)
        for i in range(2 * params['num_agents']):
            self.dist_population[i] = p[i]


