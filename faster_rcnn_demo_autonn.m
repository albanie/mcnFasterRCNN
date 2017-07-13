function faster_rcnn_demo_autonn(varargin)
%FASTER_RCNN_DEMO Minimalistic demonstration of the faster-rcnn detector
% using the dagnn wrapper

opts.modelPath = 'data/pascal/vgg16/deployed/local-faster_rcnn-pascal-14.mat' ;
opts.nmsThresh = 0.3 ;
opts.confThresh = 0.8 ;
opts.maxScale = 1000 ;
opts.scale = 600 ;

% The network is trained to prediction occurences
% of the following classes from the pascal VOC challenge
opts.classes = {'none_of_the_above', ...
    'aeroplane', ...
    'bicycle', ...
    'bird', ...
    'boat', ...
    'bottle', ...
    'bus', ...
    'car', ...
    'cat', ...
    'chair', ...
    'cow', ...
    'diningtable', ...
    'dog', ...
    'horse', ...
    'motorbike', ...
    'person', ...
    'pottedplant', ...
    'sheep', ...
    'sofa', ...
    'train', ...
    'tvmonitor'} ;
opts = vl_argparse(opts, varargin) ;

% Load the network and put it in test mode.
out = load(opts.modelPath) ; net = Net(out) ;

% Load test image
imPath = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/misc/000456.jpg') ;
im = single(imread(imPath)) ;

% resize to meet the standard faster r-cnn size criteria
imsz = [size(im,1) size(im,2)] ; maxSc = opts.maxScale ; 
factor = max(opts.scale ./ imsz) ; 
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;
data = imresize(im, factor, 'bilinear') ; 

in = {'data', data, 'im_info', imInfo} ; net.eval(in, 'forward') ;
probs = squeeze(net.getValue('cls_prob')) ;
deltas = squeeze(net.getValue('bbox_pred')) ;
props = net.getValue('proposal') ; 
boxes = props(2:end,:)' / imInfo(3) ;

% Visualize results for one class at a time
for ii = 2:numel(opts.classes)
  cprobs = probs(ii,:) ;
  cdeltas = deltas(4*(ii-1)+(1:4),:)' ;

  cboxes = bbox_transform_inv(boxes, cdeltas) ;
  cls_dets = [cboxes cprobs'] ;
  cls_dets = [boxes cprobs'] ;

  keep = bbox_nms(cls_dets, opts.nmsThresh) ;
  cls_dets = cls_dets(keep, :) ;

  sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
  if numel(sel_boxes) == 0, continue ; end

  bbox_draw(im/255,cls_dets(sel_boxes,:));
  title(sprintf('Dets for class ''%s''', opts.classes{ii})) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end

  fprintf('Detections for category ''%s'':\n', opts.classes{ii});
  for jj=1:size(sel_boxes,1)
    bbox_id = sel_boxes(jj,1);
    fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
            cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
            cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
            cls_dets(bbox_id,end));
  end
end
