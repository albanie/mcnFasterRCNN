function customObj = faster_rcnn_dagnn_custom
%FASTER_RCNN_DAGNN_CUSTOM helper funciton for convertin autonn net to dagnn
%  CUSTOMOBJ = FASTER_RCNN_DAGNN_CUSTOM returns a object which is designed
%    to allow the conversion of autonn-based faster-rcnn detectors back into
%    DagNN format.
%
% Copyright (C) 2017 Samuel Albanie
% Licensed under The MIT License [see LICENSE.md for details]

  customObj.support = {'vl_nnreshape', 'vl_nnproposalrpn', 'vl_nnroipool'} ;
  customObj.convert = @converter ;

% ----------------------------------------------------------
function [block,inputs,params] = converter(ins, out, params)
% ----------------------------------------------------------
  inputs = {ins{1}.name} ;
  funcType = func2str(out.func) ;
  switch funcType
    case 'vl_nnreshape'
      block = dagnn.Reshape() ;
      block.shape = ins{2} ;
    case 'vl_nnproposalrpn'
      block = dagnn.ProposalRPN() ;
      rpnOpts = {'featStride', 'postNMSTopN', 'preNMSTopN'} ;
      block = parseCells(block, ins, rpnOpts) ;
      inputs = cellfun(@(x) {x.name}, ins(1:3)) ;
    case 'vl_nnroipool'
      block = dagnn.ROIPooling() ;
      roiOpts = {'transform', 'method', 'Subdivisions'} ;
      block = parseCells(block, ins, roiOpts) ;
      inputs = cellfun(@(x) {x.name}, ins(1:2)) ;
    otherwise 
      error('%s is unsupported', funcType) ;
  end

% ------------------------------------
function x = parseCells(x, src, opts)
% ------------------------------------
  for jj = 1:numel(opts)
    opt = opts{jj} ;
    pos = find(strcmp(src, opt)) + 1 ;
    if ~isempty(pos) 
      if strcmp(opt, 'Subdivisions')
        x.subdivisions = src{pos} ; % roipooling handle naming issue
      else
        x.(opt) = src{pos} ; 
      end
    end
  end
