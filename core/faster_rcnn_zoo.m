function net = faster_rcnn_zoo(modelName)

modelNames = {
  'faster-rcnn-vggvd-pascal', ...
  'faster-rcnn-vggvd-coco'...
} ;

msg = sprintf('%s: unrecognised model', modelName) ;
assert(ismember(modelName, modelNames), msg) ;
modelDir = fullfile(vl_rootnn, 'data/models-import') ;
modelPath = fullfile(modelDir, sprintf('%s.mat', modelName)) ;

if ~exist(modelPath, 'file')
  fetchModel(modelName, modelPath) ;
end

net = dagnn.DagNN.loadobj(load(modelPath)) ;

% ---------------------------------------
function fetchModel(modelName, modelPath)
% ---------------------------------------

waiting = true ;
prompt = sprintf(strcat('%s was not found at %s\nWould you like to ', ...
            ' download it from THE INTERNET (y/n)?\n'), modelName, modelPath) ;

while waiting
  str = input(prompt,'s') ;
  switch str
    case 'y'
      % ensure target directory exists
      if ~exist(fileparts(modelPath), 'dir')
        mkdir(fileparts(modelPath)) ;
      end

      fprintf(sprintf('Downloading %s ... \n', modelName)) ;
      baseUrl = 'http://www.robots.ox.ac.uk/~albanie/models/faster_rcnn' ;
      url = sprintf('%s/%s.mat', baseUrl, modelName) ;
      urlwrite(url, modelPath) ;
      return ;
    case 'n'
      throw(exception) ;
    otherwise
      fprintf('input %s not recognised, please use `y` or `n`\n', str) ;
  end
end
