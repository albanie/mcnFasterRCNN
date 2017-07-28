function imdb = getCocoImdb(opts)
%GETCOCOIMDB coco imdb construction 
%  GETCOCOIMDB(OPTS) builds an image database for training and 
%  testing on the coco dataset
%
% Copyright (C) 2017 Samuel Albanie 
% All rights reserved.

  imdb = cocoSetup(opts) ;
  classIds = 1:numel(imdb.meta.classes) ;  
  imdb.classMap = containers.Map(imdb.meta.classes, classIds) ;
  imdb.images.ext = 'jpg' ;
  imdb.meta.sets = {'train', 'val', 'test'} ;

  % offset categories for background
  imdb.meta.classes = [{'background'} imdb.meta.classes] ;
  imdb.meta.supercategories = [{'background'} imdb.meta.supercategories] ;

% ------------------------------
function imdb = cocoSetup(opts)
% ------------------------------
  opts.dataDir = fullfile(opts.dataOpts.dataRoot, 'mscoco') ;
  switch opts.dataOpts.year
    case 2014
      imdb.sets.name = {'train', 'val', 'test'} ; imdb.sets.id = uint8([1 2 3]) ;
      trainImdb = buildSet(opts, 'train', 1) ; valImdb = buildSet(opts, 'val', 2) ;
      imdb = mergeImdbs(trainImdb, valImdb) ; % merge train and val
      if opts.imdbOpts.includeTest 
        testImdb = buildTestSet(opts, 'test', 3) ; 
        imdb = mergeImdbs(imdb, testImdb) ;
      end
      imdb.meta = trainImdb.meta ; % only one meta copy is needed

    case 2015
      assert(opts.imdbOpts.includeTest == true, '2015 only has test images') ;
      imdb.sets.name = {'test', 'test-dev'} ; imdb.sets.id = uint8([3 4]) ;
      testImdb = buildTestSet(opts, 'test', 3) ; 
      testDevImdb = buildTestSet(opts, 'test-dev', 4) ; 
      imdb = mergeImdbs(testImdb, testDevImdb) ; imdb.meta = testImdb.meta ;
    otherwise
      error('year %d not supported', opts.dataOpts.year) ;
  end

% -------------------------------------------------------------------------
function imdb = buildSet(opts, setName, setCode)
% -------------------------------------------------------------------------
  annoFile = sprintf('instances_%s%d.json', setName, opts.dataOpts.year) ;
  annoPath = fullfile(opts.dataDir, 'annotations', annoFile) ;
  fprintf('reading annotations from %s \n', annoPath) ;
  tmp = importdata(annoPath) ; cocoData = jsondecode(tmp{1}) ;
  imdb = getImageData(opts, cocoData, setName, setCode) ;
  imdb.meta.classes = {cocoData.categories.name} ;
  imdb.meta.supercategories = {cocoData.categories.supercategory} ;
  ids = [cocoData.annotations.image_id] ;
  annos_ = arrayfun(@(x) {cocoData.annotations(ids == x)}, imdb.images.id) ;

  annos = cell(1, numel(annos_)) ;
  for ii = 1:numel(annos_)
   fprintf('processing annotation %d/%d\n', ii, numel(annos_)) ;
   anno_ = annos_{ii} ; sz = imdb.images.imageSizes{ii} ;
   if ~opts.imdbOpts.conserveSpace
     newAnno.id = [anno_.id] ; newAnno.area = [anno_.area] ;
     newAnno.iscrowd = [anno_.iscrowd] ; newAnno.image_id = [anno_.image_id] ;
     newAnno.segmentation = {anno_.segmentation} ;
   end
   newAnno.classes = uint8([anno_.category_id]) ;
   if numel(anno_) > 0
     % store normalized boxes, rather than original pixel locations
     b = [anno_.bbox]' ; b = [b(:,1:2) b(:,1:2) + b(:,3:4)] ;
     newAnno.boxes = single(bsxfun(@rdivide, b, [sz([2 1]) sz([2 1])])) ; 
   else
     newAnno.boxes = [] ;
   end
   annos{ii} = newAnno ;
  end
  imdb.annotations = annos ;

% -------------------------------------------------------------------------
function imdb = buildTestSet(opts, setName, setCode)
% -------------------------------------------------------------------------
  annoFile = sprintf('image_info_%s%d.json', setName, opts.dataOpts.year) ;
  annoPath = fullfile(opts.dataDir, 'annotations', annoFile) ;
  fprintf('reading annotations from %s \n', annoPath) ;
  tmp = importdata(annoPath) ; cocoData = jsondecode(tmp{1}) ;
  pos = regexp(setName, '-dev$') ; % trim -dev suffix if present
  if ~isempty(pos), setName = setName(1:pos-1) ; end
  imdb = getImageData(opts, cocoData, setName, setCode) ;
  imdb.meta.classes = {cocoData.categories.name} ;
  imdb.meta.supercategories = {cocoData.categories.supercategory} ;

% -------------------------------------------------------------------------
function imdb = getImageData(opts, cocoData, setName, setCode) 
% -------------------------------------------------------------------------
  imdb.images.name = {cocoData.images.file_name} ;
  imdb.images.id = [cocoData.images.id] ;
  imdb.images.set = ones(1, numel(imdb.images.name)) * setCode ;
  height = [cocoData.images.height] ; width = [cocoData.images.width] ;
  imdb.images.imageSizes = arrayfun(@(x,y) {[x, y]}, height, width) ;
  imFolder = sprintf('%s%d', setName, opts.dataOpts.year) ;
  imdb.paths.image = esc(fullfile(opts.dataDir, 'images', imFolder, '%s')) ;
  paths = repmat(imdb.paths.image, numel(imdb.images.name), 1) ;
  imdb.images.paths = arrayfun(@(x) {paths(x,:)}, 1:size(paths,1)) ;

% --------------------------------------
function imdb = mergeImdbs(imdb1, imdb2) 
% --------------------------------------
  imdb.images.id = [imdb1.images.id imdb2.images.id] ;
  imdb.images.set = [imdb1.images.set imdb2.images.set] ;
  imdb.images.name = [imdb1.images.name imdb2.images.name] ;
  imdb.images.imageSizes = [imdb1.images.imageSizes imdb2.images.imageSizes] ;
  imdb.images.paths = horzcat(imdb1.images.paths, imdb2.images.paths) ;

  if isfield(imdb1, 'annotations') && isfield(imdb2, 'annotations')
    imdb.annotations = horzcat(imdb1.annotations, imdb2.annotations) ;
  elseif isfield(imdb1, 'annotations') 
    imdb.annotations = imdb1.annotations ;
  elseif isfield(imdb2, 'annotations') 
    imdb.annotations = imdb2.annotations ;
  else
    % pass 
  end

% -------------------------------------------------------------------------
function str=esc(str)
% -------------------------------------------------------------------------
  str = strrep(str, '\', '\\') ;
