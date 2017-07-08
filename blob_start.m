net_ = load('cNet.mat') ;
net = Net(net_) ;

store = load('trainBlobs.mat') ;
legend = load('data/models-import/imagenet-vgg-verydeep-16.mat') ;
dag = dagnn.DagNN.fromSimpleNN(legend) ;
lnet = Layer.fromDagNN(dag) ;
