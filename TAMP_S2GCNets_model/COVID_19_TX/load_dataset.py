import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
path = os.getcwd()


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'TX':
        data_path = os.path.join(path + '/covid_data/TX_COVID.npz')
        data = np.load(data_path)['arr_0']  # shape: (335, 251, 1); 1 dimension -- only number of hospitalizations at county level
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_topo_dataset(H_type):
    topo_data = np.load(path + '/covid_tda_data/TX' + '_001_MPGrid_Euler_characteristic_degree_sublevel_transaction_power.npz')['arr_0'] # shape (335, 50, 50)
    return topo_data
