function faster_rcnn_coco_demo(varargin)
% Faster R-CNN demo using the autonn wrapper -
% NOTE: This code assumes that the coco dataset has been downloaded

opts.scale = 600 ;
opts.maxScale = 1000 ;
opts.nmsThresh = 0.3 ;
opts.confThresh = 0.8 ;
opts.useGpu = false ;
opts.modelDir = fullfile(vl_rootnn, 'data/models-import') ;
opts.cocoImDir = 'data/datasets/mscoco/images' ;
opts.modelPath = '' ;
opts = vl_argparse(opts, varargin) ;


% Load or download an example faster-rcnn model:
modelName = 'faster-rcnn-vggvd-coco.mat' ;
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

dag = dagnn.DagNN.loadobj(load(opts.modelPath)) ; 
out = Layer.fromDagNN(dag, @faster_rcnn_autonn_custom_fn) ; net = Net(out{:}) ;
if opts.useGpu, net.move('gpu') ; else net.move('cpu') ;end
[~,labels] = getCocoLabelMap ; % load category map
imMean = dag.meta.normalization.averageImage ;

% load a sample image
imPath = fullfile(opts.cocoImDir, 'train2014/COCO_train2014_000000057870.jpg' ) ; 
im = single(imread(imPath)) ;  data = bsxfun(@minus, im, imMean) ;

% resize to meet the standard faster r-cnn size criteria
imsz = [size(data,1) size(data,2)] ; maxSc = opts.maxScale ; 
factor = max(opts.scale ./ imsz) ; 
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;
data = imresize(data, factor, 'bilinear') ; 

if opts.useGpu, data = gpuArray(data) ; end
in = {'data', data, 'im_info', imInfo} ; net.eval(in, 'forward') ;
probs = squeeze(net.getValue('cls_prob')) ;
deltas = squeeze(net.getValue('bbox_pred')) ;
props = net.getValue('proposal') ; boxes = props(2:end,:)' / imInfo(3) ;

for ii = 1:numel(labels)
  c = ii + 1 ;
  cprobs = probs(c,:) ;
  cdeltas = deltas(4*(c-1)+(1:4),:)' ;

  cboxes = bbox_transform_inv(boxes, cdeltas);
  cls_dets = [cboxes cprobs'] ;

  keep = bbox_nms(cls_dets, opts.nmsThresh) ;
  cls_dets = cls_dets(keep, :) ;

  %if strcmp(labels(ii), 'bus'), keyboard; end
  sel_boxes = find(cls_dets(:,end) >= opts.confThresh) ;
  if numel(sel_boxes) == 0, continue ; end

  bbox_draw(im/255,cls_dets(sel_boxes,:));
  title(sprintf('Dets for class ''%s''', labels{ii})) ;
  if exist('zs_dispFig', 'file'), zs_dispFig ; end

  fprintf('Detections for category ''%s'':\n', labels{ii});
  for j=1:size(sel_boxes,1)
    bbox_id = sel_boxes(j,1);
    fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
            cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
            cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
            cls_dets(bbox_id,end));
  end
end

if opts.useGpu, net.move('cpu') ; end % clean up
