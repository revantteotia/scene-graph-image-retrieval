import datasets
import torchvision
import json
import numpy as np

from torch_geometric.data import Data as Geo_Data
from torch_geometric.data import Batch as Geo_Batch

def src_mod_trgt_dict_list_to_graph_batch(dict_list):
    '''
    Takes a batch(list) of data dicts : {'source_img_data' : graph data of source img , 'mod' : data of modification string, 'target_img_data' : graph data of target img}

    Returns : 
        batched src image graphs : torch_geometric.data.Batch of src graphs
        batched mod strings 
        batched trgt image graphs : torch_geometric.data.Batch of trgt graphs

    '''
    src_graph_dict_list = []
    trgt_graph_dict_list = []
    mod_strings_list = []

    # print("Debug : src images list =", [ data['source_img_id'] for data in dict_list ])
    # print("Debug : trgt images list =", [ data['target_img_id'] for data in dict_list ])

    for data_dict in dict_list:
        src_graph_dict_list.append(data_dict['source_img_data'])
        trgt_graph_dict_list.append(data_dict['target_img_data'])
        mod_strings_list.append(data_dict['mod']['str'])

    src_graph_batched = graph_dict_list_to_graph_batch(src_graph_dict_list)
    trgt_graph_batched = graph_dict_list_to_graph_batch(trgt_graph_dict_list)

    return src_graph_batched, mod_strings_list, trgt_graph_batched

def graph_dict_list_to_graph_batch(dict_list):
    '''
    Takes a batch(list) of graph dicts : {'objects' : init node features , 'edge_index' : edge indices, 'edge_type' : edge types}

    Returns : 
        batched  graphs : torch_geometric.data.Batch of src graphs
    '''
   
    graphs_list = []

    for graph_dict in dict_list:
        graph_data = Geo_Data(x=graph_dict['objects'], edge_index=graph_dict['edge_index'], edge_type=graph_dict['edge_type'])
        
        graphs_list.append(graph_data)
        
    graph_batched = Geo_Batch.from_data_list(graphs_list)

    return graph_batched
