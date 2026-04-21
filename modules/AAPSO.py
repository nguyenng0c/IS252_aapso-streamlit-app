import numpy as np
import time
from math import e
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.feature_selection import *

def alturism(good_arr,bad_arr,good_vel,bad_vel,trans_func_shape='s'):
    trans_function = get_trans_function(trans_func_shape)
    for i in range(len(good_vel)):
        if good_vel[i]>0 and good_vel[i]<1.5:
            if np.random.random()<np.random.uniform(0.5,0.8):
                bad_arr[i]=good_arr[i]
                bad_vel[i]=good_vel[i]
                good_vel[i]=np.random.random()
                trans_value = trans_function(good_vel[i])
                if (np.random.random() < trans_value): 
                    good_arr[i] = 1
                else:
                    good_arr[i] = 0
        else:
            if np.random.random()<0.5:
                bad_arr[i]=good_arr[i]
                bad_vel[i]=good_vel[i]
                good_vel[i]=np.random.random()
                trans_value = trans_function(good_vel[i])
                if (np.random.random() < trans_value): 
                    good_arr[i] = 1
                else:
                    good_arr[i] = 0
    return good_arr,bad_arr,good_vel,bad_vel

def AAPSO(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_func_shape='s', save_conv_graph=False):
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)
    
    weight_acc = 0.98 # Trọng số ưu tiên Accuracy trong Khai thác dữ liệu

    # initialize particles
    particles = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    
    # initialize global and local best particles
    globalBestParticle = np.zeros(num_features)
    globalBestFitness = float("-inf")
    globalBestAcc = 0 # Lưu thêm Accuracy của hạt tốt nhất
    
    localBestParticle = np.zeros((num_agents, num_features))
    localBestFitness = np.full(num_agents, float("-inf"))
    velocity = np.zeros((num_agents, num_features))
    
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(
        train_data, train_label, stratify=train_label, shuffle=True, test_size=0.2
    )

    for iter_no in range(max_iter):
        weight = 1 - (e**-(1 - iter_no/max_iter))
        prev_fitness = fitness.copy()

        # Update velocity and position
        for i in range(num_agents):
            r1, r2 = np.random.random(2)
            velocity[i] = (weight * velocity[i]) + \
                          (r1 * (localBestParticle[i] - particles[i])) + \
                          (r2 * (globalBestParticle - particles[i]))
            
            for j in range(num_features):
                if np.random.random() < trans_function(velocity[i][j]):
                    particles[i][j] = 1
                else:
                    particles[i][j] = 0

        # Altruism process
        for i in range(num_agents):
            fitness[i], _ = compute_fitness(particles[i], data.train_X, data.val_X, data.train_Y, data.val_Y, weight_acc)
        
        delta_fit = np.subtract(fitness, prev_fitness)
        alturism_rank = np.argsort(delta_fit)
        
        num_altruists = int(0.3 * num_agents)
        for i in range(num_altruists):
            good_idx = alturism_rank[i]
            bad_idx = alturism_rank[-(i+1)]
            particles[good_idx], particles[bad_idx], velocity[good_idx], velocity[bad_idx] = \
                alturism(particles[good_idx], particles[bad_idx], velocity[good_idx], velocity[bad_idx])

        # Update global/local best
        for i in range(num_agents):
            current_fitness, current_acc = compute_fitness(particles[i], data.train_X, data.val_X, data.train_Y, data.val_Y, weight_acc)
            if current_fitness > localBestFitness[i]:
                localBestFitness[i] = current_fitness
                localBestParticle[i] = particles[i].copy()

            if current_fitness > globalBestFitness:
                globalBestFitness = current_fitness
                globalBestAcc = current_acc # Cập nhật accuracy tốt nhất
                globalBestParticle = particles[i].copy()

        if (iter_no + 1) % 10 == 0 or iter_no == 0:
            print(f'AAPSO Iteration - {iter_no+1}/{max_iter} | Best Fitness: {globalBestFitness:.4f} | Selected: {int(np.sum(globalBestParticle))}')

    # Trả về mask và accuracy để main.py in kết quả
    return globalBestParticle, globalBestAcc