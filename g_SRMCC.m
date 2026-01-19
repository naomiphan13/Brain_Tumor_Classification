function g = g_SRMCC(x, D, muK)
    mu = muK(1);
    K = muK(2);
    [N1, P] = size(D);
    
    Xh = [D(1:N1-1, :); ones(1, P)];
    y = D(N1, :);
    W = reshape(x, N1, K);
    
    % Calculate Scores & Probabilities (Forward Pass)
    Z = W' * Xh; 
    
    % Stability Fix for Gradient
    Z_stable = bsxfun(@minus, Z, max(Z, [], 1));
    A = exp(Z_stable);
    probs = bsxfun(@rdivide, A, sum(A, 1)); % Size: [K x P]
    
    % Create One-Hot Encoding for Ground Truth
    % This creates a matrix where the correct class is 1, others are 0
    ground_truth = full(sparse(double(y), 1:P, 1, K, P));
    
    % Calculate Gradient (Vectorized)
    % Gradient = 1/P * X * (Predictions - Truth)' + Regularization
    % The math: sum over samples of x * (prob - indicator)
    diff = probs - ground_truth;
    g = (Xh * diff') / P;
    
    % Add Regularization Gradient
    g = g(:) + mu * x;
end