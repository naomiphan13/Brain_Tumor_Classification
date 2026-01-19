function f = f_SRMCC(x,D,muK)
    mu = muK(1);
    K = muK(2);
    [N1,P] = size(D);
    Xh = [D(1:N1-1,:); ones(1,P)];
    y = D(N1,:);
    W = reshape(x,N1,K);
    
    % Calculate raw scores
    Z = W' * Xh;
    % Numerical Stability Fix
    Z_max = max(Z, [], 1);
    Z_stable = bsxfun(@minus, Z, Z_max);
    
    % Compute Log-Sum-Exp safely
    sum_exp = sum(exp(Z_stable), 1);
    log_sum_exp = Z_max + log(sum_exp);
    
    % Select the scores correspinding to the correct labels
    % Use linear indexing to grab the specific diagonal entries
    linear_ind = sub2ind(size(Z), y, 1:P);
    correct_class_scores = Z(linear_ind);
    
    % Compute Negative Log Likelihood
    f = -sum(correct_class_scores - log_sum_exp) / P;
    
    % Add Regularization
    xw1 = W(:);
    f = f + 0.5 * mu * (xw1' * xw1);
end