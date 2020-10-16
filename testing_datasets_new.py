# temp code to test datasets.py : will be removed later

import datasets
import dataset2
import datasetnew
import torchvision
import json
import numpy as np
trainset = datasetnew.CSSDataset(
    root='CSSDataset',
    split='test',
    transform=None
    )
print(trainset[0])