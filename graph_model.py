import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.nn import global_mean_pool

INIT_NODE_FEATURES_DIM = 15
HIDDEN_LAYER_DIM = 512
NUM_RELATIONS = 4

class Graph_Model(torch.nn.Module):
    def __init__(self,init_node_features_dim= INIT_NODE_FEATURES_DIM, 
                                          hidden_layer_dim= HIDDEN_LAYER_DIM, num_relations= NUM_RELATIONS):
        super(Graph_Model, self).__init__()

        self.conv1 = RGCNConv(in_channels=init_node_features_dim, out_channels=hidden_layer_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(in_channels=hidden_layer_dim, out_channels=hidden_layer_dim, num_relations=num_relations)



    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_type= graph.edge_type
        batch = graph.batch
        # print (" imputs to gcn fwd", x.shape, edge_index.shape, edge_type.shape)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        # 2. Obtain graph embedding by mean pooling
        x = global_mean_pool(x, batch)

        return x