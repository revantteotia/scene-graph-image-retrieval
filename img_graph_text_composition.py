
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import text_model
import torch_functions
from graph_model import Graph_Model

class ConCatModule(torch.nn.Module):

    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x


class GraphImgTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self,texts, embed_dim):
        super(GraphImgTextCompositionBase, self).__init__()
        # self.normalization_layer = torch_functions.NormalizationLayer(
        #     normalize_scale=4.0, learn_scale=True)
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=1.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

        # graph model
        graph_model = Graph_Model()

        self.graph_model = graph_model

        print("text[0] for init = ", texts[0])

        # text model
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim)

        
        img_model = torchvision.models.resnet18(pretrained=True)

        class GlobalAvgPool2d(torch.nn.Module):
            def forward(self, x):
                return F.adaptive_avg_pool2d(x, (1, 1))

        img_model.avgpool = GlobalAvgPool2d()
        img_model.fc = torch.nn.Sequential(torch.nn.Linear(512, embed_dim))
        self.img_model = img_model

    def extract_graph_feature(self, graphs):
        return self.graph_model(graphs)

    def extract_text_feature(self, texts):
        return self.text_model(texts)

    def extract_img_feature(self,imgs):
        return self.img_model(imgs)

    
    def compute_loss(self,
                    imgs_query,
                    graphs_query,
                    modification_texts,
                    imgs_target,
                    graphs_target,
                    soft_triplet_loss=True):
        mod_img1 = self.compose_img_graph_text(imgs_query, graphs_query,modification_texts)
        mod_img1 = self.normalization_layer(mod_img1)
        
        img2_features=self.extract_img_feature(imgs_target)
        graph2_features=self.extract_graph_feature(graphs_target)
        img2 = self.compose_img_graph(img2_features,graph2_features)
        img2 = self.normalization_layer(img2)

       
        assert (mod_img1.shape[0] == img2.shape[0] and
                mod_img1.shape[1] == img2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_img1, img2)
        else:
            return self.compute_batch_based_classification_loss_(mod_img1, img2)

    def compute_soft_triplet_loss_(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert (triplets and len(triplets) < 2000)
        return self.soft_triplet_loss(torch.cat([mod_img1, img2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_img1, img2):
        x = torch.mm(mod_img1, img2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)



class ConcatImgGraph(GraphImgTextCompositionBase):
    """Concatenation model."""

    def __init__(self, texts, embed_dim):
        super(ConcatImgGraph, self).__init__(texts, embed_dim)

        # composer
        class Composer(torch.nn.Module):
            """Inner composer class."""

            def __init__(self):
                super(Composer, self).__init__()
                self.m = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

            def forward(self, x):
                f = torch.cat(x, dim=1)
                f = self.m(f)
                return f
        
       
        self.composer = Composer()
    def compose_img_graph_text(self, imgs,graphs, texts):
        graph_features = self.extract_graph_feature(graphs)
        text_features = self.extract_text_feature(texts)
        img_features = self.extract_img_feature(imgs)
        img_graph_features=self.compose_img_graph(img_features,graph_features)

        return self.compose_img_graph_text_features(img_graph_features, text_features)

    def compose_img_graph_text_features(self, img_graph_features, text_features):
        return self.composer((img_graph_features, text_features))

    def compose_img_graph(self,img_features,graph_features):
        return self.composer((img_features,graph_features))

class TIRGImgGraph(GraphImgTextCompositionBase):

    def __init__(self, texts, embed_dim):
        super(TIRGImgGraph, self).__init__(texts, embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

        # composer
        class Composer(torch.nn.Module):
            """Inner composer class."""

            def __init__(self):
                super(Composer, self).__init__()
                self.m = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Linear(2 * embed_dim, 2 * embed_dim),
                    torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
                    torch.nn.Dropout(0.1), torch.nn.Linear(2 * embed_dim, embed_dim))

            def forward(self, x):
                f = torch.cat(x, dim=1)
                f = self.m(f)
                return f
        
       
        self.composer = Composer()

    def compose_img_graph_text(self,imgs, graphs, texts):
        graph_features = self.extract_graph_feature(graphs)
        text_features = self.extract_text_feature(texts)
        img_features = self.extract_img_feature(imgs)
        img_graph_features=self.compose_img_graph(img_features,graph_features)

        return self.compose_img_graph_text_features(img_graph_features, text_features)

    def compose_graph_text_features(self, img_graph_features, text_features):

        f1 = self.gated_feature_composer((img_graph_features, text_features))
        f2 = self.res_info_composer((img_graph_features, text_features))
        f = F.sigmoid(f1) * img_graph_features * self.a[0] + f2 * self.a[1]
        return f

    def compose_img_graph(self,img_features,graph_features):
        return self.Composer((img_features,graph_features))
