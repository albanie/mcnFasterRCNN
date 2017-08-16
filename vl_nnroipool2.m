function y = vl_nnroipool2(x, r, varargin)

  opts.subdivisions = [6,6] ;
  opts.transform = 1 ;
  opts.method = 'max' ;
  [dzdy,opts] = vl_argparseder(opts, varargin) ;

  if isempty(dzdy)
    y = vl_nnroipool(x,r, 'subdivisions', opts.subdivisions, ...
                          'transform', opts.transform, ...
                          'method', opts.method, ...
                          'method', opts.method) ;
  else
    y = vl_nnroipool(x,r, dzdy, 'subdivisions', opts.subdivisions, ...
                          'transform', opts.transform, ...
                          'method', opts.method) ;
  end
