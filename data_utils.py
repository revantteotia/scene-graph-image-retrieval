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
    src_graphs_list = []
    trgt_graphs_list = []
    mod_strings_list = []

    # print("Debug : src images list =", [ data['source_img_id'] for data in dict_list ])
    # print("Debug : trgt images list =", [ data['target_img_id'] for data in dict_list ])

    for data_dict in dict_list:
        src_graph_data = Geo_Data(x=data_dict['source_img_data']['objects'], edge_index=data_dict['source_img_data']['edge_index'], edge_type=data_dict['source_img_data']['edge_type'])
        trgt_graph_data = Geo_Data(x=data_dict['target_img_data']['objects'], edge_index=data_dict['target_img_data']['edge_index'], edge_type=data_dict['target_img_data']['edge_type'])
        mod_string = data_dict['mod']['str']
        
        src_graphs_list.append(src_graph_data)
        trgt_graphs_list.append(trgt_graph_data)
        mod_strings_list.append(mod_string)

    src_graph_batched = Geo_Batch.from_data_list(src_graphs_list)
    trgt_graph_batched = Geo_Batch.from_data_list(trgt_graphs_list)

    return src_graph_batched, mod_strings_list, trgt_graph_batched
