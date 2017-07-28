function targets = bboxTransform(predRois, gtRois)
%BBOXTRANSFORM compute bbox targets from predictions and ground truth
%  BBOXTRANSFORM(PREDROIS, GTROIS) computes the bounding box regression
%  targets determined by predictions PREDROIS and ground truth GTROIS

  W = predRois(:, 3) - predRois(:, 1) + 1.0 ;
  H = predRois(:, 4) - predRois(:, 2) + 1.0 ;
  ctrX = predRois(:, 1) + 0.5 * W ;
  ctrY = predRois(:, 2) + 0.5 * H ;

  gtW = gtRois(:, 3) - gtRois(:, 1) + 1.0 ;
  gtH = gtRois(:, 4) - gtRois(:, 2) + 1.0 ;
  gtCtrX = gtRois(:, 1) + 0.5 * gtW ;
  gtCtrY = gtRois(:, 2) + 0.5 * gtH ;

  dx = (gtCtrX - ctrX) ./ W ; dy = (gtCtrY - ctrY) ./ H ;
  dw = log(gtW ./ W) ; dh = log(gtH ./ H) ;
  targets = [dx dy dw dh] ;
