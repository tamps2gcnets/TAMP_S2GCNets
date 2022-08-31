import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.distance import pdist,squareform
from DNN import MAXCNN, MEANCNN
import numpy as np


class TAMPSTGCNCNN(nn.Module):
    def __init__(self, dim_in, dim_out, window_len, link_len, embed_dim):
        super(TAMPSTGCNCNN, self).__init__()
        self.link_len = link_len
        # for spatial graph convolution
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, link_len, dim_in, int(dim_out/4)))
        # for supra-Laplacian graph convolution
        if (dim_in - 1) % 16 == 0:
            self.window_weights_supra = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 4)))
        else:  # dim_in == dim_out
            self.window_weights_supra = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in / 2), int(dim_out / 4)))
        # for temporal graph convolution
        if (dim_in - 1) % 16 == 0:
            self.window_weights_temporal = nn.Parameter(torch.FloatTensor(embed_dim, 1, int(dim_out / 2)))
        else:
            self.window_weights_temporal = nn.Parameter(torch.FloatTensor(embed_dim, int(dim_in / 2), int(dim_out / 2)))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.T = nn.Parameter(torch.FloatTensor(window_len))
        self.cnn_mean = MEANCNN(window_len, int(dim_out / 8))
        self.cnn_max = MAXCNN(window_len, int(dim_out / 8))

    def forward(self, x, x_window, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, MPG):
        #S1: Laplacian construction
        # adj is only for (transportation) network which is fixed; however, for Ethereum blockchain network, there is no fixed adj
        node_num = node_embeddings.shape[0]
        initial_S = F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))) # (N, N)
        initial_S.fill_diagonal_(0)
        initial_S.fill_diagonal_(stay_cost) # probability to stay at the same layer
        if fixed_adj is True:
            S = torch.FloatTensor(np.array([adj[i, j] * np.exp(initial_S[i, j])
                                            for i in range(node_num) for j in range(node_num)]).reshape(node_num, node_num))
        else:
            S = F.softmax(initial_S, dim = 1)
        S = (S/torch.sum(S, dim = 1)).to(node_embeddings.device) # (N, N) which can be directly used as Laplacian
        support_set = [torch.eye(node_num).to(S.device), S]

        #S2: Laplacianlink
        for k in range(2, self.link_len):
            support_set.append(torch.mm(S, support_set[k-1]))
        supports = torch.stack(support_set, dim=0)

        #S3: supra-Laplacian (N*T, N*T) over multiplex network
        T = x_window.size(1)
        Bootstrap_num = np.random.choice(range(T), size=(3,))  # randomly select three elements in the window
        Bootstrap_num.sort()
        supra_laplacian = torch.zeros(size = (node_num* Bootstrap_num.shape[0], node_num* Bootstrap_num.shape[0])).to(S.device)
        inter_diagonal_matrix = np.zeros(shape=(node_num, node_num), dtype=np.float32)
        np.fill_diagonal(inter_diagonal_matrix, jump_cost)
        inter_diagonal_matrix = torch.FloatTensor(inter_diagonal_matrix).to(S.device)
        # layer 0 -> layer 1, ..., layer 0 -> layer L; layer 1 -> layer 2, ..., layer 2 -> layer L
        for i in range(Bootstrap_num.shape[0]):
            for j in range(Bootstrap_num.shape[0]):
                if i == j:
                    supra_laplacian[node_num * i : node_num * (i + 1), node_num * i : node_num * (i + 1)] = S
                elif j > i:
                    supra_laplacian[node_num * i : node_num * (i + 1), node_num * j : node_num * (j + 1)] = inter_diagonal_matrix

        #S4: spatial graph convolution
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool) #(N, link_len, dim_in, dim_out/4)
        bias = torch.matmul(node_embeddings, self.bias_pool) #(N, dim_out)
        x_s = torch.einsum("knm,bmc->bknc", supports, x) #(B, link_len, N, dim_in)
        x_s = x_s.permute(0, 2, 1, 3) #(B, N, link_len, dim_in)
        x_sconv = torch.einsum('bnki,nkio->bno', x_s, weights) #(B, N, dim_out/4)

        #S5: supra graph convolution
        weights_window_supra = torch.einsum('nd,dio->nio', node_embeddings, self.window_weights_supra)  #(N, dim_in, dim_out/4)
        x_window_ = x_window[:, Bootstrap_num, :, :]
        _x_window_ = x_window_.view(x_window_.size(0), -1, x_window_.size(3)) #(B, N*Bootstrap_num, dim_in)
        x_w_s = torch.einsum('bmi,mn->bni', _x_window_, supra_laplacian) #(B, N*Bootstrap_num, dim_in)
        x_w_s = x_w_s.view(x_w_s.size(0), Bootstrap_num.shape[0], node_num, -1) #(B, Bootstrap_num, N, dim_in)
        x_wconv_s = torch.einsum('bfni,nio->bfno', x_w_s, weights_window_supra) #(B, Bootstrap_num, N, dim_out/4)
        # global mean pooling
        x_wconv_s = torch.mean(x_wconv_s, dim=1)  #(B, N, dim_out/4)

        # S6: MLP
        weights_window_temporal = torch.einsum('nd,dio->nio', node_embeddings, self.window_weights_temporal)  #(N, dim_in, dim_out/2)
        x_w_t = torch.einsum('btni,nio->btno', x_window, weights_window_temporal)  #(B, T, N, dim_out/2)
        x_w_t = x_w_t.permute(0, 2, 3, 1)  #(B, N, dim_out/2, T)
        x_wconv_t = torch.matmul(x_w_t, self.T)  #(B, N, dim_out/2)

        #S7: mutli-persistence representation learning
        topo_mean_cnn = self.cnn_mean(MPG) #(B, dim_out/8)
        topo_max_cnn = self.cnn_max(MPG)  #(B, dim_out/8)
        topo_cnn = torch.cat([topo_mean_cnn, topo_max_cnn], dim=1) #(B, dim_out/4)
        x_twconv_s = torch.einsum('bno,bo->bno',x_wconv_s, topo_cnn)

        #S8: combination operation
        x_tswconv = torch.cat([x_sconv, x_twconv_s, x_wconv_t], dim = -1) + bias #(B, N, dim_out)
        return x_tswconv
