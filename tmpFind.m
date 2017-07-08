function tmpFind

imdb = load('data/pascal/standard_imdb/imdb.mat') ;
%candidates = cellfun(@(x) {check(x)}, imdb.annotations) ;
%keep = find([candidates{:}]) ;

t = find(imdb.images.year == 2007 & imdb.images.set <= 2) ;
imdb2 = struct() ;
imdb2.annotations = imdb.annotations(t) ;
imdb2.images.name = imdb.images.name ;
imdb2.images.year = imdb.images.year ;
imdb2.images.paths = imdb.images.paths ;
imdb2.images.imageSizes = imdb.images.imageSizes ;
imdb2.images.set = imdb.images.set ;
candidates = cellfun(@(x) {check(x)}, imdb2.annotations) ;
keep = find([candidates{:}]) ;
keyboard

for ii = 23:numel(keep)
  p = sprintf(imdb.images.paths{keep(ii)}, imdb.images.name{keep(ii)}) ;
  im = imagesc(imread(p)) ;
  zs_dispFig ;
end

% ------------------------
function res = check(anno)
% ------------------------
res = (numel(anno.classes) == 1) && anno.classes == 16 ;
if numel(res) > 1, keyboard ; end
if ~res, return ; end

%boxes = bsxfun(@times, anno.boxes, [600 800 600 800]) ;
if max(abs(anno.boxes - [0 0 1 1])) < 0.1
  res = 1 ;
else
  res = 0 ;
end
