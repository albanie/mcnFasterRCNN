function batchData = faster_rcnn_eval_get_batch(imdb, batch, opts, varargin)

bopts.prefetch = false ;
bopts = vl_argparse(bopts, varargin) ;

assert(numel(batch) <= 1, 'batch size <= 1 for RPN testing') ;

if numel(batch) == 0, batchData = {'data', [], 'imInfo', []} ; return ; end

imMean = opts.batchOpts.averageImage ; % [123, 117, 104]
useGpu = numel(opts.gpus) > 0 ;
imNames = imdb.images.name(batch) ;
imPathTemplates = imdb.images.paths(batch) ;
imPaths = cellfun(@(x,y) {sprintf(x, y)}, imPathTemplates, imNames) ;
imsz = double(imdb.images.imageSizes{batch}) ;
maxSc = opts.batchOpts.maxScale ; 
factor = max(opts.batchOpts.scale ./ imsz) ; 
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; imInfo = [ round(newSz) factor ] ;

if opts.batchOpts.use_vl_imreadjpeg
  %imMean = permute(imMean, [1 3 2]) ;
  args = {imPaths, ...
    'Verbose', ...
    'NumThreads', opts.batchOpts.numThreads, ...
    'Interpolation', 'bilinear', ...
    'SubtractAverage', squeeze(imMean), ...
    'CropAnisotropy', [1 1] ...
    'Resize', factor ...
  } ;

  if useGpu > 0 
    args{end+1} = {'Gpu'} ;
  end

  args = horzcat(args(1), args{2:end}) ;

  if bopts.prefetch
    vl_imreadjpeg(args{:}, 'prefetch') ;
    data = [] ;
  else
    out = vl_imreadjpeg(args{:}) ;
    data = out{1} ; 
  end
else
  im = single(imread(imPaths{1})) ;
  im = bsxfun(@minus, im, imMean) ;

  % note: be v. careful here - resizing by `factor` with bilinear resampling
  % does not produce the same result as resizing with non integer sizes.  To
  % reproduce the caffe features, use `factor` scalar rather than `newSz`:
  data = imresize(im, factor, 'bilinear') ; 
  if useGpu
    data = gpuArray(data) ;
  end
end

batchData = {'data', data, 'im_info', imInfo} ;
