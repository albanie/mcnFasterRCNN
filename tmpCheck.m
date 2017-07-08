x = store.rpn_cls_score_reshape ;
c = store.rpn_labels ;
c(c == -1) = -2 ; c(c == 0) = 2 ; c(c==-2) = 0 ;
c = permute(c, [4 3 1 2]) ;
x = permute(x, [4 3 2 1]) ;
l = vl_nnloss(x, c) ;
