import datasets
import numpy as np
import data_utils
import img_text_composition_models
from tensorboardX import SummaryWriter
import test_retrieval
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm as tqdm
import argparse
import sys
import time

# torch.set_num_threads(3)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

INIT_NODE_FEATURES_DIM = 15
HIDDEN_LAYER_DIM = 128
NUM_RELATIONS = 4

def parse_opt():
    """Parses the input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', type=str, default='')
    parser.add_argument('--comment', type=str, default='test_notebook')
    parser.add_argument('--dataset', type=str, default='css3d')
    parser.add_argument(
        '--dataset_path', type=str, default='../imgcomsearch/CSSDataset/output')
    parser.add_argument('--model', type=str, default='tirg')
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument(
        '--learning_rate_decay_frequency', type=int, default=9999999)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--num_iters', type=int, default=210000)
    parser.add_argument('--loss', type=str, default='soft_triplet')
    parser.add_argument('--loader_num_workers', type=int, default=4)
    args = parser.parse_args()
    return args

def load_dataset(opt):
    """Loads the input datasets."""
    print ('Reading dataset ', opt.dataset)

    trainset = datasets.CSSDataset(
    path='CSSDataset',
    split='train'
    )

    testset = datasets.CSSDataset(
    path='CSSDataset',
    split='test'
    )
    
    print ('trainset size:', len(trainset))
    print ('testset size:', len(testset))

    return trainset, testset


def create_model_and_optimizer(opt, texts):
    """Builds the model and related optimizer."""

    print ('Creating model and optimizer for', opt.model)
    if opt.model == 'imgonly':
        model = img_text_composition_models.SimpleModelImageOnly(
            texts, embed_dim=opt.embed_dim)
    elif opt.model == 'textonly':
        model = img_text_composition_models.SimpleModelTextOnly(
            texts, embed_dim=opt.embed_dim)
    elif opt.model == 'concat':
        model = img_text_composition_models.Concat(texts, embed_dim=opt.embed_dim)
    elif opt.model == 'tirg':
        model = img_text_composition_models.TIRG(texts, embed_dim=opt.embed_dim)
    elif opt.model == 'tirg_lastconv':
        model = img_text_composition_models.TIRGLastConv(
            texts, embed_dim=opt.embed_dim)
    else:
        print ('Invalid model', opt.model)
        print ('available: imgonly, textonly, concat, tirg or tirg_lastconv')
        sys.exit()
    
    model = model.to(device)

    # create optimizer
    params = []
    # low learning rate for pretrained layers on real image datasets
    if opt.dataset != 'css3d':
        params.append({
            'params': [p for p in model.img_model.fc.parameters()],
            'lr': opt.learning_rate
        })
        params.append({
            'params': [p for p in model.img_model.parameters()],
            'lr': 0.1 * opt.learning_rate
        })
    params.append({'params': [p for p in model.parameters()]})
    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.SGD(
        params, lr=opt.learning_rate, momentum=0.9, weight_decay=opt.weight_decay)
    return model, optimizer

def train_loop(opt, logger, trainset, testset, model, optimizer):
    """Function for train loop"""
    print ('Begin training')
    losses_tracking = {}
    it = 0
    epoch = -1
    tic = time.time()
    while it < opt.num_iters:
        epoch += 1

        # show/log stats
        print ('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                            4), opt.comment)
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print ('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

        # test
        if epoch % 3 == 1:
            tests = []
            for name, dataset in [('train', trainset), ('test', testset)]:
                t = test_retrieval.test(opt, model, dataset)
                tests += [(name + ' ' + metric_name, metric_value)
                        for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, it)
                print ('    ', metric_name, round(metric_value, 4))

        # save checkpoint
        torch.save({
            'it': it,
            'opt': opt,
            'model_state_dict': model.state_dict(),
        },
                logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

        # run trainning for 1 epoch
        model.train()
        trainloader = trainset.get_loader(
            batch_size=opt.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=opt.loader_num_workers)

        def training_1_iter(data):
            assert type(data) is list

            # to batch the graphs
            src_graph_batched, mod_strings_list, trgt_graph_batched = data_utils.src_mod_trgt_dict_list_to_graph_batch(data)

            img1 = src_graph_batched.to(device)
            mods = mod_strings_list
            img2 = trgt_graph_batched.to(device)

            # compute loss
            losses = []
            if opt.loss == 'soft_triplet':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=True)
            elif opt.loss == 'batch_based_classification':
                loss_value = model.compute_loss(
                    img1, mods, img2, soft_triplet_loss=False)
            else:
                print ('Invalid loss function', opt.loss)
                sys.exit()
            loss_name = opt.loss
            loss_weight = 1.0
            losses += [(loss_name, loss_weight, loss_value)]
            total_loss = sum([
                loss_weight * loss_value
                for loss_name, loss_weight, loss_value in losses
            ])

            print("Total Loss :", total_loss)
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss)]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            # gradient descend
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            training_1_iter(data)

            # decay learing rate
            if it >= opt.learning_rate_decay_frequency and it % opt.learning_rate_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

    print ('Finished training')

def main():
    opt = parse_opt()
    print ('Arguments:')
    for k in opt.__dict__.keys():
        print ('    ', k, ':', str(opt.__dict__[k]))

    logger = SummaryWriter(comment=opt.comment)
    print ('Log files saved to', logger.file_writer.get_logdir())
    for k in opt.__dict__.keys():
        logger.add_text(k, str(opt.__dict__[k]))

    trainset, testset = load_dataset(opt)
    model, optimizer = create_model_and_optimizer(
        opt, [t for t in trainset.get_all_texts()])

    train_loop(opt, logger, trainset, testset, model, optimizer)
    logger.close()


if __name__ == '__main__':
    main()