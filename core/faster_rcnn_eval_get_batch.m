function batchData = ssd_eval_get_batch(imdb, batch, opts, varargin)

bopts.prefetch = false ;
bopts = vl_argparse(bopts, varargin) ;

assert(numel(batch) <= 1, 'batch size <= 1 for RPN testing') ;

%fprintf('processing batch len: %d \n', numel(batch)) ; drawnow('update') ;
if numel(batch) == 0, batchData = {'data', [], 'imInfo', []} ; return ; end

imMean = opts.batchOpts.averageImage ; % [123, 117, 104]
useGpu = numel(opts.gpus) > 0 ;
imNames = imdb.images.name(batch) ;
imPathTemplates = imdb.images.paths(batch) ;
imPaths = cellfun(@(x,y) {sprintf(x, y)}, imPathTemplates, imNames) ;
imsz = double(imdb.images.imageSizes{batch}) ;
sc = opts.batchOpts.scale ; maxSc = opts.batchOpts.maxScale ; 
factor = max(sc ./ imsz) ; minScaleFactor = sc ./ min(imsz) ;
if any((imsz * factor) > maxSc), factor = min(maxSc ./ imsz) ; end
newSz = factor .* imsz ; imInfo = [ newSz minScaleFactor ] ;

%fprintf('processing get batch \n', batch(1)) ; drawnow('update') ;

if opts.batchOpts.use_vl_imreadjpeg
  args = {imPaths, ...
    'Verbose', ...
    'NumThreads', opts.batchOpts.numThreads, ...
    'Interpolation', 'bilinear', ...
    'SubtractAverage', imMean, ...
    'CropAnisotropy', [1 1] ...
    'Resize', newSz ...
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
  error('nah') ;
  data = single(zeros([opts.batchOpts.imageSize(1:2) 3 numel(batch)])) ;
  imMean = reshape(imMean, 1, 1, 3) ;
  for i = 1:numel(imPaths) 
    im = single(imread(imPaths{i})) ;

    if ~opts.fixedSizeInputs
      im = imresize(im, opts.batchOpts.imageSize(1:2), ...
      'method', 'bicubic') ;
    end
    data(:,:,:,i) = bsxfun(@minus, im, imMean) ;
  end

  if useGpu
    data = gpuArray(data) ;
  end
end

batchData = {'data', data, 'im_info', imInfo} ;
