function faster_rcnn_demo(varargin)
%FASTER_RCNN_DEMO Minimalistic demonstration of the faster-rcnn detector
% using the dagnn wrapper

opts.modelPath = '' ;
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

% Load or download an example faster-rcnn model:
modelName = 'faster-rcnn-vggvd-pascal.mat' ;
paths = {opts.modelPath, ...
         modelName, ...
         fullfile(vl_rootnn, 'data', 'models-import', modelName)} ;
ok = find(cellfun(@(x) exist(x, 'file'), paths), 1) ;

if isempty(ok)
  fprintf('Downloading the Faster R-CNN model ... this may take a while\n') ;
  opts.modelPath = fullfile(vl_rootnn, 'data/models-import', modelName) ;
  mkdir(fileparts(opts.modelPath)) ; base = 'http://www.robots.ox.ac.uk' ;
  url = sprintf('%s/~albanie/models/faster/%s', base, modelName) ;
  urlwrite(url, opts.modelPath) ;
else
  opts.modelPath = paths{ok} ;
end

% Load the network and put it in test mode.
net = load(opts.modelPath) ;
if isfield('forward') % autonn
  model = 'autonn' ;
  net = Net(net) ;
else
  model = 'dag' ;
  net = dagnn.DagNN.loadobj(net);
  net.mode = 'test' ;
end

% Load test image
imPath = fullfile(vl_rootnn, 'contrib/mcnFasterRCNN/misc/000456.jpg') ;
im = single(imread(imPath)) ;

% choose variables to track
if strcmp(model, 'dag')
  clsIdx = net.getVarIndex('cls_prob') ;
  bboxIdx = net.getVarIndex('bbox_pred') ;
  roisIdx = net.getVarIndex('rois') ;
  [net.vars([clsIdx bboxIdx roisIdx]).precious] = deal(true) ;
end

% resize to meet the standard faster r-cnn size criteria
imsz = [size(im,1) size(im,2)] ; maxSc = opts.maxScale ; 
factor = max(opts.scale ./ imsz) ; 
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;
data = imresize(im, factor, 'bilinear') ; 

if strcmp(mode, 'autonn')
  in = {'data', data, 'im_info', imInfo} ; net.eval(in, 'forward') ;
  probs = squeeze(net.getValue('cls_prob')) ;
  deltas = squeeze(net.getValue('bbox_pred')) ;
  props = net.getValue('proposal') ; boxes = props(2:end,:)' / imInfo(3) ;
else
  % run network and retrieve results
  net.eval({'data', data, 'im_info', imInfo}) ;
  probs = squeeze(net.vars(clsIdx).value) ;
  deltas = squeeze(net.vars(bboxIdx).value) ;
  boxes = net.vars(roisIdx).value(2:end,:)' / imInfo(3) ;
end

% Visualize results for one class at a time
for i = 2:numel(opts.classes)
  c = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
  cprobs = probs(c,:) ;
  cdeltas = deltas(4*(c-1)+(1:4),:)' ;

  cboxes = bbox_transform_inv(boxes, cdeltas);
  cls_dets = [cboxes cprobs'] ;

  keep = bbox_nms(cls_dets, opts.nmsThresh) ;
  cls_dets = cls_dets(keep, :) ;

  sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
  if numel(sel_boxes) == 0, continue ; end

  bbox_draw(im/255,cls_dets(sel_boxes,:));
  title(sprintf('Dets for class ''%s''', opts.classes{i})) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end

  fprintf('Detections for category ''%s'':\n', opts.classes{i});
  for j=1:size(sel_boxes,1)
    bbox_id = sel_boxes(j,1);
    fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
            cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
            cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
            cls_dets(bbox_id,end));
  end
end
