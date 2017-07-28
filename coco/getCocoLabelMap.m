function [labelMap,labels] = getCocoLabelMap(varargin)
%GETCOCOLABELMAP provides a mapping between coco category indices
% [LABELMAP,LABELS] = GETCOCOLABELMAP returns a Map object `LABELMAP`
% with keys 1 to K where K is number of MS COCO classes (currently 80) 
% and theircorresponding `category ids` as defined in the COCO paper.  
% `LABELS` is a cell array containing the names of each category
%
% GETCOCOLABELMAP takes the following options:
% `labelMapFile` :: path
%   The path to a csv file containing they label mapping
% `reverse` :: false
%   reverse the map direction, so that it maps from category ids to class
%   indicies
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  opts.reverse = false ;
  opts.labelMapFile = fullfile(vl_rootnn, 'data/coco/label_map.txt') ;
  opts = vl_argparse(opts, varargin) ;

  data = importdata(opts.labelMapFile) ; 
  tokens = cellfun(@(x) {strsplit(x, ',')}, data) ;
  keys = cellfun(@(x) str2num(x{2}), tokens) ;
  vals = cellfun(@(x) str2num(x{1}), tokens) ;
  labels = cellfun(@(x) {x{3}}, tokens) ;
  if opts.reverse, tmp = keys ; keys = vals ; vals = tmp ; end 
  labelMap = containers.Map(keys, vals) ;
