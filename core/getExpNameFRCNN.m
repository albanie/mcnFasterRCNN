function expName = getExpNameFRCNN(mopts, dopts) 
%GETEXPNAMEFRCNN create an appropriate experiment name  
%   GETEXPNAMEFRCNN(MOPTS, DOPTS) defines a naming strategy for each 
%   experiment, based on the model options MOPTS and data options DOPTS
%   provided
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  if dopts.useValForTraining, subset = 'vt' ; else, subset = 't' ; end
  args = {mopts.type, dopts.name, subset, dopts.trainData, mopts.batchSize, ...
         mopts.architecture} ;
  expName = sprintf('%s-%s-%s-%s-%d-%s', args{:}) ; 
  if dopts.flipAugmentation, expName = [ expName '-flip' ] ; end
  if dopts.patchAugmentation, expName = [ expName '-patch' ] ; end
  if dopts.distortAugmentation, expName = [ expName '-distort' ] ; end
  if dopts.zoomAugmentation
      expName = [ expName sprintf('-zoom-%d', dopts.zoomScale) ] ;
  end
  if mopts.instanceNormalization, expName = [ expName '-in' ] ; end
