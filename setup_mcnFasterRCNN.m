function setup_mcnFasterRCNN()
%SETUP_MCNFASTERRCNN Sets up mcnFasterRCNN, by adding its folders 
% to the Matlab path

  root = fileparts(mfilename('fullpath')) ;
  addpath(root) ;
  addpath(root, [root '/matlab'], [root '/pascal'], [root '/core'] ) ;
  addpath([root '/pascal/helpers']) ;

% temp fix
faster_startup ; addpath([root '/scale_analysis']) ;
