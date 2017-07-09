rois = (store.rois + 1)' ;
feats = permute(store.conv5_3, [3 4 2 1]) ;
noise = 1e-1 ;
feats = feats + randn(size(feats)) * noise ;

res = vl_nnroipool(feats, rois, 'Subdivisions', [7,7], 'transform', 1/16) ;
cRes = permute(store.pool5, [3 4 2 1]) ;
diff = norm(cRes(:) - res(:)) / norm(res(:)) ;
fprintf('diff %.2f (noise: %g)\n', diff, noise) ;
