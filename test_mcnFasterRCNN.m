function test_mcnFasterRCNN
% run tests for Faster R-CNN module
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_faster_rcnn_tests('command', 'nn') ;
