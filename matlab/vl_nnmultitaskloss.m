function [y, dzdl] = vl_nnmultitaskloss(x, l, varargin)
%VL_NNMULTITASKLOSS computes the Multtask Loss
%   Y = VL_NNMULTITASKLOSS(X, L) computes the Multitask Loss which 
%   is a weighted combination of the class prediction loss and 
%   boudning box regression loss for an r-cnn style object detector. The
%   inputs X, L and N are scalars where X is the class prediction loss,
%   L is the bounding box regression loss. The output Y is a SINGLE scalar.
%
%   The Multitask Loss produced by X and L is:
%     Loss = X + LOCWEIGHT * L
%   where `LOCWEIGHT` is a scalar weighting term (described below).
%
%   DERS = VL_NNMULTITASKLOSS(X, L, DZDY) computes the derivatives 
%   with respect to inputs X and L, where DERS = {DZDX, DZDL}. Here
%   DZDX, DZDL and DZDY have the same dimensions as X, L and Y respectively.
%
%   VL_NNMULTIBOXLOSS(..., 'option', value, ...) takes the following option:
%
%   `locWeight`:: 1
%    A scalar which weights the loss contribution of the regression loss. 

opts.locWeight = 1 ;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;
keyboard

if isempty(dzdy)
    y = x + opts.locWeight * l ;
else
    dzdx = dzdy{1} ; 
    dzdl = dzdx * opts.locWeight ;
    y = dzdx ;
end
