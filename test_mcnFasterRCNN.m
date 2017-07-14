function test_mcnFasterRCNN
% ----------------------------------
% run tests for Faster R-CNN module
% ----------------------------------

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_faster_rcnn_tests('command', 'nn') ;

% test utils
run_faster_rcnn_tests('command', 'ut') ;
