function newBoxes = anchorCoder(boxes, to)
%ANCHORCODER converts anchor boxes between formats
%   NEWBOXES = ANCHORCODER(BOXES,TO) converts a set of boxes between
%   two formats, defined below. BOXES is an Nx4 array in either of the
%   formats described below, TO is a string declaring the target format
%   (can be 'CenWH' or 'anchor') and NEWBOXES is a Nx4 array containing
%   the converted boxes. Box formats:
%
%   'CenWH': [CENTERX CENTERY WIDTH HEIGHT]
%   'anchor': [XMIN YMIN WIDTH HEIGHT]
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  switch to
    case 'CenWH'
      WH = boxes(:,3:4,:,:) - boxes(:,1:2,:,:) + 1 ;
      CenXY = boxes(:,1:2,:,:) + (WH -1) / 2 ;
      newBoxes = [CenXY WH] ;
    case 'anchor'
      newBoxes = [boxes(:,1) - 0.5 .* (boxes(:,3) -1) ...
                  boxes(:,2) - 0.5 .* (boxes(:,4) -1) ...
                  boxes(:,1) + 0.5 .* (boxes(:,3) -1) ...
                  boxes(:,2) + 0.5 .* (boxes(:,4) -1)] ;
    otherwise, error('%s is not a supported encoding', to) ;
  end
