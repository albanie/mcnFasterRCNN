classdef ProposalRPN < dagnn.ElementWise
  properties
    featStride = 16
    scales = [8 16 32]
    fixed = [] 
    postNMSTopN = 300 
    preNMSTopN = 6000 
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnproposalrpn(inputs{1}, inputs{2}, inputs{3}, ...
                       'featStride', obj.featStride, 'scales', obj.scales, ...
                       'fixed', obj.fixed) ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ; derParams = {} ;
    end
    
    function rfs = getReceptiveFields(obj)
    end
    
    function load(obj, varargin)
      s = dagnn.Layer.argsToStruct(varargin{:}) ;
      load@dagnn.Layer(obj, s) ;
    end
    
    function obj = PriorBox(varargin)
      obj.load(varargin{:}) ;
      obj.minSize = obj.minSize ;
      obj.maxSize = obj.maxSize ;
      obj.aspectRatios = obj.aspectRatios ;
      obj.flip = obj.flip ;
      obj.clip = obj.clip ;
      obj.variance = obj.variance ;
    end
  end
end
