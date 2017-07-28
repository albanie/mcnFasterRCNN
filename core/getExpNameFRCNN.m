function expName = getExpNameFRCNN(mopts, dopts) 
%GETEXPNAMEFRCNN defines a naming strategy for each experiment, 
% depending on the options used during training

if dopts.useValForTraining, subset = 'vt' ; else, subset = 't' ; end
args = {mopts.type, dopts.name, subset, mopts.batchSize, mopts.architecture} ;
expName = sprintf('%s-%s-%s-%d-%s', args{:}) ; 

if dopts.flipAugmentation, expName = [ expName '-flip' ] ; end
if dopts.patchAugmentation, expName = [ expName '-patch' ] ; end
if dopts.distortAugmentation, expName = [ expName '-distort' ] ; end

if dopts.zoomAugmentation
    expName = [ expName sprintf('-zoom-%d', dopts.zoomScale) ] ;
end
