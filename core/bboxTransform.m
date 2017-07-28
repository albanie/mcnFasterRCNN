function predBoxes = bboxTransform(rois, gtBoxes)

  W = rois(:,3) - rois(:,1) + 1 ;
  H = rois(:,4) - rois(:,2) + 1 ;
  ctrX = rois(:,1) + 0.5 * W ;
  ctrY = rois(:,2) + 0.5 * H ;

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

  dX = deltas(:,1:4:end) ; dY = deltas(:,2:4:end) ;
  dW = deltas(:,3:4:end) ; dH = deltas(:,4:4:end) ;
  predCtrX = bsxfun(@plus, bsxfun(@times, dX,W), ctrX) ;
  predCtrY = bsxfun(@plus, bsxfun(@times, dY,H), ctrY) ;
  keyboard
  predW = bsxfun(@times, exp(dW), W) ;
  predH = bsxfun(@times, exp(dH), H) ;

  predBoxes = zeros(size(deltas)) ;
  predBoxes(:,1:4:end) = predCtrX - 0.5 * predW ;
  predBoxes(:,2:4:end) = predCtrY - 0.5 * predH ;
  predBoxes(:,3:4:end) = predCtrX + 0.5 * predW ;
  predBoxes(:,4:4:end) = predCtrY + 0.5 * predH ;

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
