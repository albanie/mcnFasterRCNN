function batchData = faster_rcnn_train_get_batch(imdb, batch, bopts, varargin)
% SSD_TRAIN_GET_BATCH generates mini batches for training faster r-cnn, with 
% the option to apply SSD-style data augmentation

  imNames = imdb.images.name(batch) ; imPathTemplates = imdb.images.paths(batch) ;
  imPaths = cellfun(@(x,y) sprintf(x, y), imPathTemplates, imNames, 'Uni', 0) ;
  annotations = imdb.annotations(batch) ;
  targets = cellfun(@(x) {x.boxes}, annotations) ;
  labels = cellfun(@(x) {single(x.classes)}, annotations) ;

  imsz = double(imdb.images.imageSizes{batch}) ;
  maxSc = bopts.maxScale ; factor = max(bopts.scale ./ imsz) ; 
  if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
  newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ; 
  data = single(zeros([imInfo(1:2) 3 numel(batch)])) ; 
  RGB = [122.7717, 115.9465, 102.9801] ; % follow girshick here
  imMean = permute(RGB, [3 1 2]) ;

  % randomly generate resize methods
  resizers = bopts.resizers(randi(numel(bopts.resizers), 1, numel(batch))) ;

  for i = 1:numel(batch)
    im = single(imread(imPaths{i})) ; sz = size(im) ;
    targets_ = targets{i} ; labels_ = labels{i} ; 

    if bopts.distortOpts.use % apply image distortion
      if rand < bopts.distortOpts.brightnessProb
        delta = bopts.distortOpts.brightnessDelta ;
        assert(delta >= 0, 'brightness delta must be non-negative') ;
        adjust = -delta + rand * 2 * delta ; % adjust brightness and clip
        im = max(min(im + adjust, 255), 0) ;
      end

      if rand < bopts.distortOpts.contrastProb
        lower = bopts.distortOpts.contrastLower ;
        upper = bopts.distortOpts.contrastUpper ;
        assert(upper >= lower, 'upper contrast must be >= lower') ;
        assert(lower >= 0, 'lower contrast must be non-negative') ;
        adjust = lower + rand * (upper - lower) ;% adjust contrast and clip
        im = max(min(im * adjust, 255), 0) ;
      end

      if rand < bopts.distortOpts.saturationProb
        lower = bopts.distortOpts.saturationLower ;
        upper = bopts.distortOpts.saturationUpper ;
        assert(upper >= lower, 'upper saturation must be >= lower') ;
        assert(lower >= 0, 'lower saturation must be non-negative') ;
        adjust = lower + rand * (upper - lower) ; im_ = rgb2hsv(im / 255) ;
        sat = max(min(im_(:,:,2) * adjust,  1), 0) ; % adjust sat & clip
        im_(:,:,2) = sat ; im = hsv2rgb(im_) * 255 ;
      end

      if rand < bopts.distortOpts.hueProb
        delta = bopts.distortOpts.hueDelta ;
        assert(delta >= 0, 'hue delta must be non-negative') ;
        adjust = -delta + rand * 2 * delta ; im_ = rgb2hsv(im / 255) ;
        hue = max(min(im_(:,:,1) + adjust,  1), 0) ; % adjust hue and clip
        im_(:,:,1) = hue ; im = hsv2rgb(im_) * 255 ;
      end

      if rand < bopts.distortOpts.randomOrderProb
        im = im(:,:,randperm(3)) ;
      end
    end

    if bopts.zoomOpts.use && rand < bopts.zoomOpts.prob % zoom out
      minScale = bopts.zoomOpts.minScale ; maxScale = bopts.zoomOpts.maxScale ;
      zoomScale = minScale + rand * (maxScale - minScale) ;
      canvasSize = [ round(sz(1:2) * zoomScale) 3 ] ;
      canvas = bsxfun(@times, ones(canvasSize), permute(RGB, [3 1 2])) ;
      minYX_ = rand(1, 2) ; % uniformly sample location from feasible region
      minYX = minYX_ .* (canvasSize(1:2) - sz(1:2)) ;
      yLoc = round(minYX(1) + 1:minYX(1) + sz(1)) ;
      xLoc = round(minYX(2) + 1:minYX(2) + sz(2)) ;
      canvas(yLoc,xLoc,:) = im ; % insert image at location

      % update targets
      targetsMinWH = bboxCoder(targets_, 'MinMax', 'MinWH') ;
      updatedWH = targetsMinWH(:,3:4) / zoomScale ;
      offsets = [minYX_(2) minYX_(1)] * (zoomScale - 1 ) / zoomScale ;
      updatedXY = bsxfun(@plus, offsets, targetsMinWH(:,1:2) / zoomScale) ;
      updatedTargetsMinWH = [updatedXY updatedWH ] ;
      targets_ = bboxCoder(updatedTargetsMinWH, 'MinWH', 'MinMax') ;
      im = canvas ; sz = size(im) ;
    end

    if bopts.patchOpts.use % sample a patch
      [patch, targets_, labels_] = patchSampler(targets_, labels_, bopts.patchOpts) ;
      xmin = 1 + round(patch(1) * (sz(2) - 1)) ;
      xmax = 1 + round(patch(3) * (sz(2) - 1)) ;
      ymin = 1 + round(patch(2) * (sz(1) - 1)) ;
      ymax = 1 + round(patch(4) * (sz(1) - 1)) ;
      im = im(ymin:ymax, xmin:xmax, :) ;
    end

    im = bsxfun(@minus, im, imMean) ; % match caffe pre-proc order
    im = imresize(im, imInfo(1:2), 'method', resizers{i}) ;  % num check

    if bopts.flipOpts.use && rand < bopts.flipOpts.prob % flipping
      im = fliplr(im) ;
      targets_ = [1 - targets_(:,3) targets_(:,2) 1 - targets_(:,1) targets_(:,4)] ;
    end

    labels{i} = labels_ ; 
    % scale targets back to pixel space - work with 0-index for consistency 
    % with python implementation
    offset = imInfo(3) ; % single scaled pixel
    targets{i} = bsxfun(@times, targets_, imInfo([2 1 2 1])) - offset ; 
    data(:,:,:,i) = im ;
  end

  if bopts.useGpu, data = gpuArray(data) ; end
  %if bopts.debug, viz_get_batch(data, labels, targets) ; end

  if bopts.debug
    tmp = load('contrib/mcnFasterRCNN/feats/input_img.mat') ;
    labels{1} = tmp.gt_boxes(5) ;
    keyboard
  end
    %imPaths = {

  batchData = {'data', data, 'gtLabels', labels, ...
               'gtBoxes', targets, 'imInfo', imInfo} ;
