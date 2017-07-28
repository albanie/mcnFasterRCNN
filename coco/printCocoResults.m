function res = printCocoResults(cacheDir, varargin)

opts.expTag = '' ;
opts.orientation = 'portrait' ;
opts = vl_argparse(opts, varargin) ;

% handle either a single, or cell array of inputs
if isstr(cacheDir), cacheDirs = {cacheDir} ; else cacheDirs = cacheDir ; end

valList = selectExp(cacheDir, opts) ;
res = cellfun(@(x) getResult(x), valList) ;                     

%--------------------------------------------------------
function [valList, testList] = selectExp(cacheDirs, opts) 
%--------------------------------------------------------
valList = cellfun(@getResultsList, cacheDirs, 'Uni', 0) ;
if isempty(opts.expTag), valList = [valList{:}] ; return ; end % no tag, return all
keep = [] ; % otherwise, return selected
for i = 1:numel(testList)
  s = testList{i} ; fname = sprintf('%s.mat', opts.expTag) ;
  if ~isempty(s) && exist(fullfile(fileparts(s{1}), fname, 'file')) ;
    keep(end+1) = i ;
  end
end
valList = [valList{keep}] ;

% -------------------------------------------------------------------------
function [val, test] = getResultsList(cacheDir)
% -------------------------------------------------------------------------
files = ignoreSystemFiles(dir(fullfile(cacheDir, '*.mat'))) ; 
names = {files.name} ;
[set, sfx] = cellfun(@(x) getSuffixes(x), names, 'Uni', false) ;
val = fullfile(cacheDir, names(strcmp(sfx, 'results') & strcmp(set, 'val'))) ;

% -------------------------------------------------------------------------
function [penSuffix, suffix] = getSuffixes(filename) 
% -------------------------------------------------------------------------
[~,filename,~] = fileparts(filename) ;
tokens = strsplit(filename, '-') ;
if numel(tokens) == 1 
  penSuffix = '' ; suffix = '' ;
else
  penSuffix = tokens{end -1} ; suffix = tokens{end} ;
end

% -------------------------------------------------------------------------
function result = getResult(resultFile)
% -------------------------------------------------------------------------
[~,fname,~] = fileparts(resultFile) ;
tokens = strsplit(fname,'-') ;
model = strjoin(tokens(1:end - 2), '-') ;
result.model = model;
result.subset = tokens{end - 1} ;
data = load(resultFile) ;
fprintf('---\n%s\n---\n', model) ;
ev = CocoEval() ; ev.eval = data.results ; ev.summarize() ;
a = 1 ; m = 3 ; s = data.results.precision(:,:,:,a,m) ;
result = mean(s(s>=0)) ;
