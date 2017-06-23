% This script evaluates the pre-trained models released by Ross Girshick

% The evaluation is done on the VOC 2007 test data. 
evalVersion = 'fast' ; % switch to `official` for submissions

models = {...
    'faster-rcnn-pascal-vggvd', ...
}

for i = 1:numel(models)
    model = models{i} ;
    faster_rcnn_pascal_evaluation('modelName', model, ...
                                  'evalVersion', evalVersion) ;
end

