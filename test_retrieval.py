# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates the retrieval model."""
import numpy as np
import torch
from tqdm import tqdm as tqdm
import data_utils

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# temp list of empty images : to ignore images with no objects
empty_images_list=[('test', 9832), ('train', 1565), ('test', 4132), ('train', 2881), ('train', 1161), ('train', 9313), ('test', 17076), ('train', 4363), ('train', 1221), ('test', 7235), ('test', 12701), ('train', 2446), ('train', 9837), ('test', 11441), ('test', 1349), ('train', 3023), ('train', 14329), ('test', 3918), ('train', 11780), ('train', 4729), ('train', 15256), ('train', 6460), ('test', 5471), ('test', 12318), ('train', 2821), ('test', 11197), ('test', 1100), ('test', 14355), ('train', 3155), ('train', 7540), ('test', 18320), ('train', 6668), ('train', 16116), ('test', 5346), ('test', 1275), ('train', 1087), ('test', 6223), ('train', 10547), ('test', 13383)]


def test(opt, model, testset, split='test'):
    """Tests a model over the given testset."""
    model.eval()
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        # compute test query features
        imgs = []
        mods = []
        for t in tqdm(test_queries):

            # if src or target in empty list then continue
            if ((split,t['source_img_id']) in empty_images_list) or ((split,t['target_caption']) in empty_images_list):
                continue

            imgs += [testset.get_scene(t['source_img_id'])] # list of graph dicts
            mods += [t['mod']['str']]
            all_target_captions += [t['target_caption']]

            if len(imgs) >= opt.batch_size or t is test_queries[-1]: # batch the images and mods
                # if 'torch' not in str(type(imgs[0])):
                #     imgs = [torch.from_numpy(d).float() for d in imgs]
                # imgs = torch.stack(imgs).float()
                # imgs = torch.autograd.Variable(imgs).to(device)
                # mods = [t.decode('utf-8') for t in mods]

                batched_src_graphs = data_utils.graph_dict_list_to_graph_batch(imgs)
                imgs = batched_src_graphs.to(device)

                f = model.compose_graph_text(imgs, mods).data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        # all_target_captions = [t['target_caption'] for t in test_queries]

        # compute all image features
        imgs = []
        for i in tqdm(range(len(testset.imgs))):

            # if src or target in empty list then continue
            if (split,i) in empty_images_list:
                continue

            imgs += [testset.get_scene(i)]
            all_captions += [testset.imgs[i]['captions'][0]]

            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                # if 'torch' not in str(type(imgs[0])):
                #     imgs = [torch.from_numpy(d).float() for d in imgs]
                # imgs = torch.stack(imgs).float()
                # imgs = torch.autograd.Variable(imgs).to(device)

                batched_graphs = data_utils.graph_dict_list_to_graph_batch(imgs)
                imgs = batched_graphs.to(device)

                imgs = model.extract_graph_feature(imgs).data.cpu().numpy()
                all_imgs += [imgs]
                imgs = []
        all_imgs = np.concatenate(all_imgs)
        # all_captions = [img['captions'][0] for img in testset.imgs]

    else:
        # use training queries to approximate training retrieval performance
        imgs0 = []
        imgs = []
        mods = []
        for i in range(10000):
            item = testset[i]

            # if src or target in empty list then continue
            if ((split,item['source_img_id']) in empty_images_list) or ((split,item['target_caption']) in empty_images_list):
                continue            

            imgs += [item['source_img_data']]
            mods += [item['mod']['str']]
            if len(imgs) > opt.batch_size or i == 9999:
                # imgs = torch.stack(imgs).float()
                # imgs = torch.autograd.Variable(imgs)
                # mods = [t.decode('utf-8') for t in mods]

                batched_src_graphs = data_utils.graph_dict_list_to_graph_batch(imgs)
                imgs = batched_src_graphs.to(device)

                f = model.compose_graph_text(imgs.to(device), mods).data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
            imgs0 += [item['target_img_data']]
            if len(imgs0) > opt.batch_size or i == 9999:
                # imgs0 = torch.stack(imgs0).float()
                # imgs0 = torch.autograd.Variable(imgs0)

                batched_graphs = data_utils.graph_dict_list_to_graph_batch(imgs0)
                imgs0 = batched_graphs.to(device)

                imgs0 = model.extract_graph_feature(imgs0.to(device)).data.cpu().numpy()
                all_imgs += [imgs0]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]
        all_imgs = np.concatenate(all_imgs)
        all_queries = np.concatenate(all_queries)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    for k in [1, 5, 10, 50, 100]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', r)]

        if opt.dataset == 'mitstates':
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_adj', r)]

            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_noun', r)]

    return out
