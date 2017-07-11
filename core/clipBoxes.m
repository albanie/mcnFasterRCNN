function boxes = clipBoxes(boxes, imsz)
% -------------------------------------

boxes(:,1:4:end) = max(min(boxes(:,1:4:end),imsz(2)-1),0);
boxes(:,2:4:end) = max(min(boxes(:,2:4:end),imsz(1)-1),0);
boxes(:,3:4:end) = max(min(boxes(:,3:4:end),imsz(2)-1),0);
boxes(:,4:4:end) = max(min(boxes(:,4:4:end),imsz(1)-1),0);
