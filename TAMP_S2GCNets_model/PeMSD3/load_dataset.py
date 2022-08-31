import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
path = os.getcwd()


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    elif dataset == 'PEMSD3':
        data_path = os.path.join(path + '/data/PEMS03/PEMS03.npz')
        data = np.load(data_path)['data'][:, :, 0]  # one dimensions, traffic flow data
        print(data.shape)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_topo_dataset(data_type):
    topo_data = np.load(path + '/tda_data/PEMSD' + data_type + '_01_MPGrid_Euler_characteristic_degree_sublevel_betweenness_sublevel.npz')['arr_0']

    return topo_data
