classdef nnsmoothL1loss < nntest
  methods (Test)

    function basic(test)
      x = test.randn([100 1]) ;
      t = test.randn([100 1]) ;
      y = vl_nnsmoothL1loss(x, t) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsmoothL1loss(x, t, dzdy) ;
      test.der(@(x) vl_nnsmoothL1loss(x, t), x, dzdy, dzdx, 1e-3*test.range) ;
    end

    function basicWithInstanceWeights(test)
      x = test.randn([100 1]) ;
      t = test.randn([100 1]) ;

      iw = test.randn([100 1]) ; 
      ow = test.randn([100 1]) ; 
      optargs = {'insideWeights', iw, 'outsideWeights', ow} ;
      y = vl_nnsmoothL1loss(x, t, optargs{:}) ;

      % check derivatives with numerical approximation
      dzdy = test.randn(size(y)) ;
      dzdx = vl_nnsmoothL1loss(x, t, dzdy, optargs{:}) ;
      test.der(@(x) vl_nnsmoothL1loss(x, t, optargs{:}), ...
                                         x, dzdy, dzdx, 1e-3*test.range) ;
    end

  end
end
