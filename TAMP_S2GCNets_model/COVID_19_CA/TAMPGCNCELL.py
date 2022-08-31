import torch
import torch.nn as nn
import torch.nn.functional as F
from TAMPGCN import TAMPSTGCNCNN

class TAMPGCRNCNNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, window_len, link_len, embed_dim):
        super(TAMPGCRNCNNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = TAMPSTGCNCNN(dim_in+self.hidden_dim, 2*dim_out, window_len, link_len, embed_dim)
        self.update = TAMPSTGCNCNN(dim_in+self.hidden_dim, dim_out, window_len, link_len, embed_dim)

    def forward(self, x, state, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, MPG):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1) #x + state
        z_r = torch.sigmoid(self.gate(input_and_state, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, MPG))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, x_full, node_embeddings, fixed_adj, adj, stay_cost, jump_cost, MPG))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
