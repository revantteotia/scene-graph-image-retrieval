import torch
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv
from torch_geometric.nn import global_mean_pool
import datasets
import numpy as np

import data_utils


INIT_NODE_FEATURES_DIM = 15
HIDDEN_LAYER_DIM = 128
NUM_RELATIONS = 4

class BasicRGCN(torch.nn.Module):
    def __init__(self):
        super(BasicRGCN, self).__init__()
        self.conv1 = RGCNConv(in_channels=INIT_NODE_FEATURES_DIM, out_channels=HIDDEN_LAYER_DIM, num_relations=NUM_RELATIONS)
        self.conv2 = RGCNConv(in_channels=HIDDEN_LAYER_DIM, out_channels=HIDDEN_LAYER_DIM, num_relations=NUM_RELATIONS)



    def forward(self, x, edge_index, edge_type, batch):
        print (x.shape, edge_index.shape, edge_type.shape)
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        # 2. Obtain graph embedding by mean pooling
        x = global_mean_pool(x, batch)  

        return x

model = BasicRGCN()
print(model)


### TESTING MODEL ON SINGLE DATA POINT

trainset = datasets.CSSDataset(
    path='CSSDataset',
    split='train'
    )

trainloader = trainset.get_loader(
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=1)

##-----------

def extract_graph_features(model, graph_data):
    x = graph_data.x
    edge_index = graph_data.edge_index
    edge_type = graph_data.edge_type
    batch = graph_data.batch

    print("x shape", x.shape)

    graph_embedding = model(x=x, edge_index=edge_index, edge_type=edge_type, batch=batch)

    return graph_embedding


def compute_loss(src_graphs_batched, mod_strings, trgt_graphs_batched):
    '''
    TEMP FUNCTION : WILL BE CALLED BY composition model
    
    Extracts features for batches of graph data and mod texts --> creates composition --> calculates triplet loss
    '''
    loss = 0

    src_features = extract_graph_features(model, src_graphs_batched)
    trgt_features = extract_graph_features(model, trgt_graphs_batched)

    print("src_features shape", src_features.shape)
    print("trgt_features shape", trgt_features.shape)


    print("to calculate triplet loss using src feat, target feat, text feat")

    return loss

def training_1_iter(data):
    '''
    TEMP FUNCTION : WILL BE CALLED BY MAIN METHOD
    
    Batches the list of data dicts and then calls compute loss
    '''
    assert type(data) is list

    src_graphs_batched, mod_strings_list, trgt_graphs_batched = data_utils.src_mod_trgt_dict_list_to_graph_batch(data)

    print(src_graphs_batched)
    print(trgt_graphs_batched)
    print(mod_strings_list)

    print("now compute loss using batched graphs")

    loss = compute_loss(src_graphs_batched, mod_strings_list, trgt_graphs_batched)
    print(loss)

for i, data in enumerate(trainloader):
    training_1_iter(data)    
    if i==0:
        break


exit()
##----------------



# first_data_point = trainset[0]

# # -- CALCULATING SRC GRAPH EMB
# x_src = first_data_point['source_img_data']['objects']
# x_src = torch.Tensor(x_src)
# print('x_src= \n', x_src)

# edge_index_src = first_data_point['source_img_data']['edge_index']
# edge_index_src = torch.from_numpy(edge_index_src.astype(np.long))
# print('edge_index_src = \n', edge_index_src)

# edge_types_src = first_data_point['source_img_data']['edge_types']
# edge_types_src = torch.from_numpy(edge_types_src.astype(np.long))
# print('edge_types_src = \n', edge_types_src)


# # testing fwd pass on model
# graph_embedding_src = model(x_src, edge_index_src, edge_types_src, batch = torch.zeros(x_src.shape[0], dtype=torch.long)) # all nodes of 1 batch

# print("graph_embedding_src shape", graph_embedding_src.shape)

# print("graph_embedding_src : ", graph_embedding_src)

# # -- CALCULATING TRGT GRAPH EMB

# x_trgt = first_data_point['target_img_data']['objects']
# x_trgt = torch.Tensor(x_trgt)
# print('x_trgt= \n', x_trgt)

# edge_index_trgt = first_data_point['target_img_data']['edge_index']
# edge_index_trgt = torch.from_numpy(edge_index_trgt.astype(np.long))
# print('edge_index_trgt = \n', edge_index_trgt)

# edge_types_trgt = first_data_point['target_img_data']['edge_types']
# edge_types_trgt = torch.from_numpy(edge_types_trgt.astype(np.long))
# print('edge_types_trgt = \n', edge_types_trgt)


# # testing fwd pass on model
# graph_embedding_trgt = model(x_trgt, edge_index_trgt, edge_types_trgt, batch = torch.zeros(x_trgt.shape[0], dtype=torch.long)) # all nodes of 1 batch

# print("graph_embedding_trgt shape", graph_embedding_trgt.shape)

# print("graph_embedding_trgt : ", graph_embedding_trgt)

