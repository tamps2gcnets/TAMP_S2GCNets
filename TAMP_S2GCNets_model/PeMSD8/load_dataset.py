import os
import numpy as np
path = os.getcwd()


def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join(path + '/data/PEMS04/PEMS04.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join(path + '/data/PEMS08/PEMS08.npz')
        data = np.load(data_path)['data'][:, :, 0:3]  #three dimensions, traffic flow data
        print(data.shape)
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data


def load_topo_dataset(data_type):
    topo_data = np.load(path + '/tda_data/PEMSD' + data_type + '_01_MPGrid_Euler_characteristic_betweenness_sublevel_attribute_power.npz')['arr_0']

    return topo_data
