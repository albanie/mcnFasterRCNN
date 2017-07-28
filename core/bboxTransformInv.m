function predBoxes = bboxTransformInv(boxes, deltas)
%BBOXTRANSFORMINV invert bbox target encoding
%  BBOXTRANSFORMINV(BOXES, DELTAS) computes the boxes predicted by 
%  applying DELTAS to the given set of BOXES under the inverse bbox
%  parameterisation introduced in the paper:
%
%    Girshick, Ross, et al. "Rich feature hierarchies for accurate object 
%    detection and semantic segmentation." Proceedings of the IEEE 
%    conference on computer vision and pattern recognition. 2014.

  W = boxes(:,3) - boxes(:,1) + 1 ;
  H = boxes(:,4) - boxes(:,2) + 1 ;
  ctrX = boxes(:,1) + 0.5 * W ;
  ctrY = boxes(:,2) + 0.5 * H ;

  dX = deltas(:,1:4:end) ; dY = deltas(:,2:4:end) ;
  dW = deltas(:,3:4:end) ; dH = deltas(:,4:4:end) ;
  predCtrX = bsxfun(@plus, bsxfun(@times, dX,W), ctrX) ;
  predCtrY = bsxfun(@plus, bsxfun(@times, dY,H), ctrY) ;
  predW = bsxfun(@times, exp(dW), W) ;
  predH = bsxfun(@times, exp(dH), H) ;

  predBoxes = zeros(size(deltas), 'like', deltas) ;
  predBoxes(:,1:4:end) = predCtrX - 0.5 * predW ;
  predBoxes(:,2:4:end) = predCtrY - 0.5 * predH ;
  predBoxes(:,3:4:end) = predCtrX + 0.5 * predW ;
  predBoxes(:,4:4:end) = predCtrY + 0.5 * predH ;

  %widths = anchors(:,3) - anchors(:,1) + 1 ;
  %heights = anchors(:,4) - anchors(:,2) + 1 ;
  %ctrX = anchors(:,1) + 0.5 .* widths ;
  %ctrY = anchors(:,2) + 0.5 .* heights ;

  %dx = bboxDeltas(:,1) ;
  %dy = bboxDeltas(:,2) ;
  %dw = bboxDeltas(:,3) ;
  %dh = bboxDeltas(:,4) ;

  %predCenX = dx .* widths + ctrX ;
  %predCenY = dy .* heights + ctrY ;
  %predW = exp(dw) .* widths ; 
  %predH = exp(dh) .* heights ;
  %proposals = [ predCenX - 0.5 * predW ...
            %predCenY - 0.5 * predH ...
            %predCenX + 0.5 * predW ...
            %predCenY + 0.5 * predH] ;
