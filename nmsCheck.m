rng(0) ;
opts.useGpu = 0 ;
numBoxes = 100 ;
boxes = rand(numBoxes, 5) ;
boxes(:,3:4) = bsxfun(@minus, 1, boxes(:,1:2)) .* boxes(:,3:4) + boxes(:,1:2) ;
overlap = single(0.3) ; boxes = single(boxes) ;

if opts.useGpu
  boxes = gpuArray(boxes) ; overlap = gpuArray(overlap) ; 
else
  boxes = gather(boxes) ; overlap = gather(overlap) ;
end
pick1 = bbox_nms(boxes, overlap) ;
boxes2 = boxes' ;
bb = bboxCoder(boxes, 'MinMax', 'MinWH') ; areas = prod(bb(:,[3 4]), 2) ;
%disp(areas) ;

keyboard
inBoxes = single(boxes)' ;
%inBoxes = single(1) ;
pick2 = vl_nnbboxnms(inBoxes, overlap) ;

fprintf('num pick1 %d \n', numel(pick1)) ;
fprintf('num pick2 %d \n', numel(pick2)) ;
