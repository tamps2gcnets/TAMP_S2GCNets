import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
path = os.getcwd()

# parameter setting
alpha = 0.1
NVertices = 358 # Number of vertices

data_path = os.path.join(path + '/PEMS03.npz')
PEMS_features = np.load(data_path)['data'] #one dimensions, traffic flow data; shape is (26208, 358, 1)

PEMS_nodes_indices = pd.read_csv(path + '/PEMS03_nodes_indices.csv', header=0).values

PEMS_net_dataset = pd.read_csv(path + '/PEMS03_edgelist.csv', header=0)
PEMS_net_edges = PEMS_net_dataset.values[:, 0:2]
PEMS_net_edgelist_ = [(int(u), int(v)) for u, v in PEMS_net_edges] # the num of nodes is 358, the number of edges is 547
PEMS_net_edgelist = [(np.where(PEMS_nodes_indices[:,0] == u)[0][0], np.where(PEMS_nodes_indices[:,0] == v)[0][0]) for u,v in PEMS_net_edgelist_]

PEMS_networks_list = []
for i in range(PEMS_features.shape[0]): # 26208 # PEMS_features.shape[0]
    PEMS_networks = np.zeros(shape=(len(PEMS_net_edgelist), 3), dtype=np.float32) # from (node id), to (node id), weight
    PEMS_networks[:, 0:2] = np.array(PEMS_net_edgelist)
    tmp_features = PEMS_features[i, :, :].reshape(NVertices, -1)
    for j in range(len(PEMS_net_edgelist)):
        u, v = PEMS_net_edgelist[j]
        if np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2)) == 0:
            PEMS_networks[j, 2] = 1e-5
        else:
            PEMS_networks[j, 2] = np.sqrt(np.sum((tmp_features[int(u), :] - tmp_features[int(v), :]) ** 2))

    tmp_max = np.max(PEMS_networks[:, 2])
    PEMS_networks[:, 2] = PEMS_networks[:, 2] / tmp_max

    # cut edges
    tmp_cut_pair = np.where(PEMS_networks[:, 2] > alpha)
    PEMS_networks[tmp_cut_pair[0], 2] = 0.
    PEMS_networks_list.append(PEMS_networks)

np.save('PEMS_networks_list', PEMS_networks_list)

