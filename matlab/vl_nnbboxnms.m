%VL_NNBBOXNMS Non-maximum supression
%   Y = VL_NNBBOXNMS(B, OVERLAP) greedily selects the collection of 
%   boxes corresponding to the highest scores that have not yet been 
%   covered by previous selections. 
%
%  To keep things fast, the input shape is 5xn rather than the nx5 commonly
%  used.  Moreover, it is assumed that the boxes are sorted in order of 
%  descending scores (where scores are stored in the fifth row).
