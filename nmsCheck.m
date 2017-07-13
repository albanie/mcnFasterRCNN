rng(2) ;
sc = 100 ;
opts.disp = 0 ;
opts.useGpu = 1 ;
numBoxes = 50000 ;
boxes = rand(numBoxes, 4) ;
boxes(:,3:4) = bsxfun(@minus, 1, boxes(:,1:2)) .* boxes(:,3:4) + boxes(:,1:2) ;
overlap = single(0.3) ; 
boxes = [single(round(boxes * sc)) rand(numBoxes, 1) ] ;

% NOTE: tied scores may not be broken the same way

if opts.useGpu
  boxes = gpuArray(boxes) ;
else
  boxes = gather(boxes) ; overlap = gather(overlap) ;
end

[~,sIdx] = sort(boxes(:,5), 'descend') ;
boxes = boxes(sIdx,:) ;

tic ; pick1 = bbox_nms(boxes, overlap) ; t1 = toc ;

bb = bboxCoder(boxes, 'MinMax', 'MinWH') ; 
areas = prod(bb(:,[3 4]), 2) ;
%disp(areas) ;

inBoxes = single(boxes)' ;
tic ; pick2 = vl_nnbboxnms(inBoxes, overlap) ; t2 = toc ;

fprintf('t1: %g\n', t1) ;
fprintf('t2: %g\n', t2) ;

fprintf('num pick1 %d \n', numel(pick1)) ;
fprintf('num pick2 %d \n', numel(pick2)) ;

% 
b = gather(boxes(:,1:4)) ;
bWH = bboxCoder(b(pick1,:), 'MinMax', 'MinWH') ;

if opts.disp
  clf ; figure ;
  for ii = 1:size(bWH,1) 
    rectangle('Position', bWH(ii,:), 'EdgeColor', 'Red') ;
    hold on ;
  end
  xlim([0 100]) ; ylim([0 100]) ;
  zs_dispFig ;
  ov = bbox_overlap(b,b) ;
  %pp = triu(ov,1) ; disp(pp(pp<overlap)') ;
end
