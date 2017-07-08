im = permute(store.data, [3 4 2 1]) ; im = im(:,:,[3 2 1]) ;
mini = lnet{1}.find('relu1_1', 1) ;

mini_f = mini.find('conv1_1', 1).inputs{2}.value
mini_b = mini.find('conv1_1', 1).inputs{3}.value

miniNet = Net(mini) ;
in = {'x0', im} ;
miniNet.eval(in, 'forward') ;
out = miniNet.getValue('relu1_1') ;
