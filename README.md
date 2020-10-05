# scene-graph-image-retrieval
To implement the idea of using scene-graphs for text based image retrieval.

Inspired from [Composing Text and Image for Image Retrieval - An Empirical Odyssey](https://arxiv.org/abs/1812.07119)

Code is largely based on [github.com/google/tirg](https://github.com/google/tirg)



#### TODO:
* Code is in python 2.7. Convert it to python 3.6 while coding.
* Find a way to add object location in its initial representation

* Use [Pytorch Geometric RGCN](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.RGCNConv), \([example](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/rgcn.py)\) with scene graph info to get visual representation

* Understanding following things from TIRG code [github.com/google/tirg](https://github.com/google/tirg)
    * How is loss calculated
    * How is the model evaluated
    * How is composition of textual and visual feature done

