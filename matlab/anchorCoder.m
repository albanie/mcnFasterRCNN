function newBoxes = anchorCoder(boxes, to)

switch to
    case 'CenWH'
        WH = boxes(:,3:4,:,:) - boxes(:,1:2,:,:) + 1 ;
        CenXY = boxes(:,1:2,:,:) + (WH -1) / 2 ;
        newBoxes = [ CenXY WH] ;
    case 'anchor'
        newBoxes = [boxes(:,1) - 0.5 .* (boxes(:,3) -1) ...
                    boxes(:,2) - 0.5 .* (boxes(:,4) -1) ...
                    boxes(:,1) + 0.5 .* (boxes(:,3) -1) ...
                    boxes(:,2) + 0.5 .* (boxes(:,4) -1)] ;
    otherwise
        fprintf('%s is not a supported encoding', to) ;
end
