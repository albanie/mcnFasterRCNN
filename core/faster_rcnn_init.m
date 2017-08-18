function net = faster_rcnn_init(opts, varargin)
% FASTER_RCNN_INIT Initialize a Faster R-CNN Detector Network
%   FASTER_RCNN_INIT(OPTS) - constructs a Faster R-CNN detector 
%   according to the options provided using the autonn matconvnet
%   wrapper.

  rng(0) ; % for reproducibility, fix the seed

  % select RPN/fast-rcnn locations
  modelName = opts.modelOpts.architecture ;
  switch modelName 
    case 'vgg16'
      rpn.base = 'relu5_3' ;
      frcn.base = 'relu5_3' ; rpn.channels_in = 512 ;
      rpn.channels_out = 512 ;
      frcn.channels_in = 4096 ;
      frcn.heads = {'fc6'} ;
      frcn.insert_dropout = true ;
      frcn.divisions = [7, 7] ;
      frcn.tail = 'drop7' ;
      freeze_decay = 1 ;
    case 'resnet50'
      rpn.base = 'res4f_relu' ;
      frcn.base = 'res4f_relu' ;
      rpn.channels_in = 1024 ;
      rpn.channels_out = 256 ;
      frcn.heads = {'res5a_branch1', 'res5a_branch2a'} ;
      frcn.channels_in = 2048 ;
      frcn.divisions = [7, 7] ;
      frcn.insert_dropout = false ;
      frcn.tail = 'pool5' ;
      if opts.modelOpts.resDoubleStride
         frcn.subdivisions = frcn.subdivisions * 2 ;
      end
      freeze_decay = 0 ;
  end

  % configure autonn inputs
  gtBoxes = Input('gtBoxes') ; 
  gtLabels = Input('gtLabels') ; 
  imInfo = Input('imInfo') ;

  % freeze early layers and modify trunk biases to match caffe
  dag = trunk_model_zoo(modelName, opts) ;
  dag = freeze_and_match_layers(dag, freeze_decay, opts) ;

  % convert to autonn
  stored = Layer.fromDagNN(dag) ; net = stored{1} ;

  % pass relevant inputs to RPN
  rpn.gtBoxes = gtBoxes ; rpn.imInfo = imInfo ; rpn.gtLabels = gtLabels ;
  rpn_out = attach_rpn(net, rpn, opts) ;
  det_out = attach_frcn(net, frcn, rpn_out, opts) ;
  net = compile_detector(rpn_out, det_out, dag.meta, opts) ;
