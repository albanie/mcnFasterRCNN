function expName = getExpNameFRCNN(mopts, dopts) 
%GETEXPNAMEFRCNN create an appropriate experiment name  
%   GETEXPNAMEFRCNN(MOPTS, DOPTS) defines a naming strategy for each 
%   experiment, based on the model options MOPTS and data options DOPTS
%   provided
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  if has(dopts, 'useValForTraining'), subset = 'vt' ; else, subset = 't' ; end
  args = {mopts.type, dopts.name, subset, dopts.trainData, mopts.batchSize, ...
         mopts.architecture} ;
  expName = sprintf('%s-%s-%s-%s-%d-%s', args{:}) ; 
  if has(dopts, 'flipAugmentation'), expName = [ expName '-flip' ] ; end
  if has(dopts, 'patchAugmentation'), expName = [ expName '-patch' ] ; end
  if has(dopts, 'distortAugmentation'), expName = [ expName '-distort' ] ; end
  if has(mopts, 'atrous'), expName = [ expName '-atrous' ] ; end
  if has(dopts, 'zoomAugmentation')
      expName = [ expName sprintf('-zoom-%d', dopts.zoomScale) ] ;
  end
  if has(mopts, 'instanceNormalization'), expName = [ expName '-in' ] ; end

% --------------------------
function res = has(s, key)
% ---------------------------
% check for both existence and truth of value
  res = isfield(s, key) && s.(key) ;
