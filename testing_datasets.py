# temp code to test datasets.py : will be removed later

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
    
    for data_dict in dict_list:
        src_graph_data = Geo_Data(x=data_dict['source_img_data']['objects'], edge_index=data_dict['source_img_data']['edge_index'], edge_types=data_dict['source_img_data']['edge_types'])
        trgt_graph_data = Geo_Data(x=data_dict['target_img_data']['objects'], edge_index=data_dict['target_img_data']['edge_index'], edge_types=data_dict['target_img_data']['edge_types'])
        mod_string = data_dict['mod']['str']
        
        src_graphs_list.append(src_graph_data)
        trgt_graphs_list.append(trgt_graph_data)
        mod_strings_list.append(mod_string)

    src_graph_batched = Geo_Batch.from_data_list(src_graphs_list)
    trgt_graph_batched = Geo_Batch.from_data_list(trgt_graphs_list)

    return src_graph_batched, mod_strings_list, trgt_graph_batched


trainset = datasets.CSSDataset(
    path='CSSDataset',
    split='test',
    transform=None)

trainloader = trainset.get_loader(
        batch_size=2,
        shuffle=True,
        drop_last=True,
        num_workers=1)



for step, batch in enumerate(trainloader):
    print(f'Step {step + 1}:')
    print('=======')
    print('data in the current batch', len(batch), type(batch))
    # print(batch)
    print()
 
    src_graph_batched, mod_strings_list, trgt_graph_batched = src_mod_trgt_dict_list_to_graph_batch(batch)

    print(src_graph_batched, src_graph_batched.batch)
    print(trgt_graph_batched)
    print(mod_strings_list)
    if step == 0:
        break


# print (type(data))

# print(len(trainset))

# all_texts = trainset.get_all_texts()

# print(type(all_texts))


# first_data_point = trainset[0]
# # test_queries = trainset.get_test_queries()
# print( 'src img id = \n',  first_data_point['source_img_id'])
# print( 'src img data X = \n',  first_data_point['source_img_data']['objects'])
# print( 'src img data edge_index = \n', first_data_point['source_img_data']['edge_index'].shape,  first_data_point['source_img_data']['edge_index'])
# print( 'src img data edge_types = \n',  first_data_point['source_img_data']['edge_types'])

# print( 'trgt img id = \n',  first_data_point['target_img_id'])
# print( 'trgt img X = \n',  first_data_point['target_img_data']['objects'])
# print( 'trgt img edge_index = \n',  first_data_point['target_img_data']['edge_index'].shape, first_data_point['target_img_data']['edge_index'])
# print( 'trgt img edge_types = \n',  first_data_point['target_img_data']['edge_types'])

# print( 'modification string = \n',  first_data_point['mod']['str'])
# print(test_queries[0])

