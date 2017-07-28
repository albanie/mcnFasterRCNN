function predBoxes = bboxTransformInv(boxes, deltas)

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

  predBoxes = zeros(size(deltas)) ;
  predBoxes(:,1:4:end) = predCtrX - 0.5 * predW ;
  predBoxes(:,2:4:end) = predCtrY - 0.5 * predH ;
  predBoxes(:,3:4:end) = predCtrX + 0.5 * predW ;
  predBoxes(:,4:4:end) = predCtrY + 0.5 * predH ;
