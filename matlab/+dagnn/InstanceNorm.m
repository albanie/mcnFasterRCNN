classdef InstanceNorm < dagnn.ElementWise
  properties
    numChannels
    epsilon = 1e-5
    opts = {'NoCuDNN'} % ours seems slightly faster
  end

  properties (Transient)
    moments
  end

  methods
    function outputs = forward(obj, inputs, params)
        % Essentially, instance norm works by fixing batch norm to work
        % in train mode without moments
        outputs{1} = vl_nnbnorm(inputs{1}, params{1}, params{2}, ...
                      'epsilon', obj.epsilon, ...
                      obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [derInputs{1}, derParams{1}, derParams{2}] = ...
        vl_nnbnorm(inputs{1}, params{1}, params{2}, derOutputs{1}, ...
                   'epsilon', obj.epsilon, ...
                   obj.opts{:}) ;
    end

    % ---------------------------------------------------------------------
    function obj = InstanceNorm(varargin)
      obj.load(varargin{:}) ;
    end

    function params = initParams(obj)
      params{1} = ones(obj.numChannels,1,'single') ;
      params{2} = zeros(obj.numChannels,1,'single') ;
      params{3} = zeros(obj.numChannels,2,'single') ;
    end

    function attach(obj, net, index)
      attach@dagnn.ElementWise(obj, net, index) ;
      p = net.getParamIndex(net.layers(index).params{3}) ;
      net.params(p).trainMethod = 'average' ;
      net.params(p).learningRate = 0.1 ;
    end
  end
end
