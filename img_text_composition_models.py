
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


class graphTextCompositionBase(torch.nn.Module):
    """Base class for image + text composition."""

    def __init__(self):
        super(graphTextCompositionBase, self).__init__()
        # self.normalization_layer = torch_functions.NormalizationLayer(
        #     normalize_scale=4.0, learn_scale=True)
        self.normalization_layer = torch_functions.NormalizationLayer(
            normalize_scale=1.0, learn_scale=True)
        self.soft_triplet_loss = torch_functions.TripletLoss()

    def extract_graph_feature(self, graphs):
        raise NotImplementedError

    def extract_text_feature(self, texts):
        raise NotImplementedError

    def compose_graph_text(self, graphs, texts):
        raise NotImplementedError

    def compute_loss(self,
                    graphs_query,
                    modification_texts,
                    graphs_target,
                    soft_triplet_loss=True):
        mod_graph1 = self.compose_graph_text(graphs_query, modification_texts)
        mod_graph1 = self.normalization_layer(mod_graph1)
        graph2 = self.extract_graph_feature(graphs_target)
        graph2 = self.normalization_layer(graph2)

        # TO DEBUG
        # if (torch.sum(torch.isnan(mod_graph1)) or torch.sum(torch.isnan(graph2))):
        #     print("FOOBAR")
        #     print("mod_graph1 :", mod_graph1)
        #     print("graph12 :", graph2)
        #     assert False
        # print("Graph1 Graph2 shapes : ", mod_graph1.shape, graph2.shape)
        
        assert (mod_graph1.shape[0] == graph2.shape[0] and
                mod_graph1.shape[1] == graph2.shape[1])
        if soft_triplet_loss:
            return self.compute_soft_triplet_loss_(mod_graph1, graph2)
        else:
            return self.compute_batch_based_classification_loss_(mod_graph1, graph2)

    def compute_soft_triplet_loss_(self, mod_graph1, graph2):
        triplets = []
        labels = list(range(mod_graph1.shape[0])) + list(range(graph2.shape[0]))
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
        return self.soft_triplet_loss(torch.cat([mod_graph1, graph2]), triplets)

    def compute_batch_based_classification_loss_(self, mod_graph1, graph2):
        x = torch.mm(mod_graph1, graph2.transpose(0, 1))
        labels = torch.tensor(range(x.shape[0])).long()
        labels = torch.autograd.Variable(labels).cuda()
        return F.cross_entropy(x, labels)


class graphEncoderTextEncoderBase(graphTextCompositionBase):
    """Base class for image and text encoder."""

    def __init__(self, texts, embed_dim):
        super(graphEncoderTextEncoderBase, self).__init__()

        # graph model
        graph_model = Graph_Model()

        self.graph_model = graph_model

        print("text[0] for init = ", texts[0])

        # text model
        self.text_model = text_model.TextLSTMModel(
            texts_to_build_vocab=texts,
            word_embed_dim=embed_dim,
            lstm_hidden_dim=embed_dim)

    def extract_graph_feature(self, graphs):
        return self.graph_model(graphs)

    def extract_text_feature(self, texts):
        return self.text_model(texts)


class SimpleModelGraphOnly(graphEncoderTextEncoderBase):

    def compose_graph_text(self, graphs, texts):
        return self.extract_graph_feature(graphs)


class SimpleModelTextOnly(graphEncoderTextEncoderBase):

    def compose_graph_text(self, graphs, texts):
        return self.extract_text_feature(texts)


class Concat(graphEncoderTextEncoderBase):
    """Concatenation model."""

    def __init__(self, texts, embed_dim):
        super(Concat, self).__init__(texts, embed_dim)

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

    def compose_graph_text(self, graphs, texts):
        graph_features = self.extract_graph_feature(graphs)
        text_features = self.extract_text_feature(texts)
        return self.compose_graph_text_features(graph_features, text_features)

    def compose_graph_text_features(self, graph_features, text_features):
        return self.composer((graph_features, text_features))


class TIRG(graphEncoderTextEncoderBase):
    """The TIGR model.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim):
        super(TIRG, self).__init__(texts, embed_dim)

        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))
        self.res_info_composer = torch.nn.Sequential(
            ConCatModule(), torch.nn.BatchNorm1d(2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, 2 * embed_dim), torch.nn.ReLU(),
            torch.nn.Linear(2 * embed_dim, embed_dim))

    def compose_graph_text(self, graphs, texts):
        graph_features = self.extract_graph_feature(graphs)
        text_features = self.extract_text_feature(texts)

        print("graph_features shape", graph_features.shape)
        print("text_features shape", text_features.shape)
        return self.compose_graph_text_features(graph_features, text_features)

    def compose_graph_text_features(self, graph_features, text_features):
        f1 = self.gated_feature_composer((graph_features, text_features))
        f2 = self.res_info_composer((graph_features, text_features))
        f = F.sigmoid(f1) * graph_features * self.a[0] + f2 * self.a[1]
        return f


class TIRGLastConv(graphEncoderTextEncoderBase):
    """The TIGR model with spatial modification over the last conv layer.

    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, texts, embed_dim):
        super(TIRGLastConv, self).__init__(texts, embed_dim)
        
    def compose_graph_text(self, graphs, texts):
        x = graphs
        
        return x
