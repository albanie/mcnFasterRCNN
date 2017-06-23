imPath = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/python/000456.jpg') ;
data = single(imread(imPath)) ;
im_info = [size(data, 1) size(data,2) 1.6] ;
dag.eval({'data', data, 'im_info', im_info}) ;

net = dag ;

opts.classes = {'background', ...
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', ...
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', ...
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'} ;

confThresh = 0.8 ; nmsThresh = 0.3 ;
probs = squeeze(dag.vars(dag.getVarIndex('cls_prob')).value) ;
deltas = squeeze(dag.vars(dag.getVarIndex('bbox_pred')).value) ;
boxes = dag.vars(dag.getVarIndex('rois')).value(2:end,:)' ; % / im_info(3) ;


% Visualize results for one class at a time
for i = 2:numel(opts.classes)
  c = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
  cprobs = probs(c,:) ;
  cdeltas = deltas(4*(c-1)+(1:4),:)' ;

  cboxes = bbox_transform_inv(boxes, cdeltas);
  cls_dets = [cboxes cprobs'] ;

  keep = bbox_nms(cls_dets, nmsThresh) ;
  cls_dets = cls_dets(keep, :) ;

  sel_boxes = find(cls_dets(:,end) >= confThresh) ;
  if numel(sel_boxes) == 0, continue ; end
    keyboard

  imo = bbox_draw(data/255,cls_dets(sel_boxes,:));
  title(sprintf('Detections for class ''%s''', opts.classes{i})) ;
  if exist('zs_dispFig'), zs_dispFig ; end

  fprintf('Detections for category ''%s'':\n', opts.classes{i});
  for j=1:size(sel_boxes,1)
    bbox_id = sel_boxes(j,1);
    fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
            cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
            cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
            cls_dets(bbox_id,end));
  end
end
