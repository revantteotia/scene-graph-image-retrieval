# temp code to test datasets.py : will be removed later

import datasets
import torchvision
import json
import numpy as np
trainset = datasets.CSSDataset(
    path='CSSDataset',
    split='test',
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])
    ]))

first_data_point = trainset[0]
# test_queries = trainset.get_test_queries()
print( 'src img id = \n',  first_data_point['source_img_id'])
print( 'src img data = \n',  first_data_point['source_img_data'])
print( 'trgt img id = \n',  first_data_point['target_img_id'])
print( 'trgt img data = \n',  first_data_point['target_img_data'])
print( 'modification string = \n',  first_data_point['mod']['str'])
# print(test_queries[0])