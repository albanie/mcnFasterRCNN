function setup_mcnFasterRCNN()
%SETUP_MCNFASTERRCNN Sets up mcnFasterRCNN, by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(root, [root '/matlab'], [root '/pascal'], [root '/core']) ;
  addpath([root '/pascal/helpers'], [root '/coco'], [root '/misc']) ;
  addpath([vl_rootnn '/examples/fast_rcnn/bbox_functions']) ;

  % (only needed for tuning purposes)
  addpath([root '/train_analysis']) ;
