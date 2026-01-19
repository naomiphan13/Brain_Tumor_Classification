%  Program: bt_lsearch2019.m
%  Implements line learch by backtracking.
%  Description: Implements inexact line search described in 
%  Reference: Sec. 9.2 of Boyd and Vanderberghe's book.
%  Input:
%     x:  initial point
%     d:  search direction
% fname:  objective function to be minimized along the direction of s  
% gname:  gradient of the objective function.
%    p1:  user-defined parameter vector whose components mast have been 
%         numerically specified. The order in which the components of p1 
%         appear must be the same as what they appear in fname and gname.
%         Note: p1 is an optional input.
%    p2:  2nd user-defined parameter vector whose components mast have been 
%         numerically specified. The order in which the components of p2 
%         appear must be the same as what they appear in fname and gname.
%         Note: p2 is an optional input.
% Output:
%     a:  acceptable value of alpha.
% Written by W.-S. Lu, University of Victoria.
% Last modified: July 28, 2019.
function a = bt_lsearch2019(x, d, f_handle, g_handle, D, muK)
    % Inputs f_handle and g_handle must be function handles (e.g., @f_SRMCC)
    rho = 0.1;
    gma = 0.5;
    
    x = x(:);
    d = d(:);
    a = 1;
    xw = x + a*d;
    
    % Direct call (No EVAL)
    f0 = f_handle(x, D, muK);
    g0 = g_handle(x, D, muK);
    f1 = f_handle(xw, D, muK);
    
    t0 = rho * (g0' * d);
    f2 = f0 + a * t0;
    er = f1 - f2;
    
    while er > 0
         a = gma * a;
         xw = x + a * d;
         
         % Direct call inside loop
         f1 = f_handle(xw, D, muK);
         
         f2 = f0 + a * t0;
         er = f1 - f2;
    end
    
    if a < 1e-5
       a = min([1e-5, 0.1 / (norm(d) + eps)]); 
    end 
end