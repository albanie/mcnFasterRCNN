function y = vl_nnsmoothL1loss(x, t, varargin)
%VL_NNSMOOTHL1LOSS computes the Huber Loss
%   Y = VL_NNSMOOTHL1LOSS(X, T) computes the smooth L1 Loss (also known 
%   as the huber loss) between an N x 1 array of inputs 
%   predictions. The output Y is a SINGLE scalar.
%
%   The smooth L1 Loss between X and T is as:
%
%     Loss = sum(f(X - T) .* W)
%
%   where W is an aray of instance weights (described below) and f is 
%   the following function (using the Faster R-CNN definition):
%   
%          { 0.5 * sigma^2 * x^2,    if |x| < 1 / sigma^2
%   f(x) = {
%          { |x| - 0.5 / sigma^2,    otherwise.
%
%   DZDX = VL_NNSMOOTHL1LOSS(X, T, DZDY) computes the derivatives with 
%   respect to inputs X. DZDX and DZDY have the same dimensions
%   as X and Y respectively.  The derivative of the Huber Loss is 
%   computed using
%
%          { sigma^2 * x,      if |x| < 1 / sigma^2,
%  f'(x) = {
%          { sign(x),          otherwise.
%
%   VL_NNSMOOTHL1LOSS(..., 'option', value, ...) takes the following option:
%
%   `insideWweights`:: []
%    If given, weights the distance between x and t *before* computing the 
%    value of the smoothL1 loss funciton.
%
%   `outsideWweights`:: []
%    If given, weights the distance between x and t *after* computing the 
%    value of the smoothL1 loss funciton.
%
% Copyright (C) 2017 Samuel Albanie, Hakan Bilen
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.sigma = 1 ;
opts.insideWeights = [] ;
opts.outsideWeights = [] ;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

delta = x - t ;
if ~isempty(opts.insideWeights), delta = delta .* opts.insideWeights ; end
absDelta = abs(delta) ; sigma2 = opts.sigma ^ 2 ;
linearRegion = (absDelta > 1. / sigma2) ;

if isempty(dzdy)
    absDelta(linearRegion) = absDelta(linearRegion) - 0.5 / sigma2 ;
    absDelta(~linearRegion) = 0.5 * sigma2 * absDelta(~linearRegion) .^ 2 ;
    if ~isempty(opts.outsideWeights)
      y = opts.outsideWeights(:)' * absDelta(:) ;
    else
      y = sum(absDelta(:)) ;
    end
else
    delta(linearRegion) = sign(delta(linearRegion));
    delta(~linearRegion) = sigma2 * delta(~linearRegion) ;
    if ~isempty(opts.outsideWeights), delta = delta .* opts.outsideWeights ; end
    if ~isempty(opts.insideWeights), delta = delta .* opts.insideWeights ; end
    y = delta .* dzdy{1} ;
end
