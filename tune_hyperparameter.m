function [best_mu, best_accuracy] = tune_hyperparameter(Xtr, ytr, grid, K_folds, K, iter)
    cv = cvpartition(ytr(:), 'KFold', K_folds);
    results = zeros(length(grid), 2);

    for i = 1:length(grid)
        mu = grid(i);
        fprintf('Calculating mu: %.4f\n', mu);
        fold_accuracy = zeros(K_folds, 1);

        % 5-Fold Cross Validation
        for k = 1:K_folds
            % Get Fold Data
            trainIdx = cv.training(k);
            valIdx = cv.test(k);
            
            X_train_fold = Xtr(:, trainIdx);
            y_train_fold = ytr(trainIdx);
            X_val_fold = Xtr(:, valIdx);
            y_val_fold = ytr(valIdx);
            
            % --- TRAIN ---
            % Pass the current_mu to the training function
            [Ws, ~] = SRMCC_bfgsML([X_train_fold; y_train_fold], ...
                                   'f_SRMCC', 'g_SRMCC', mu, K, iter); 
            
            % --- PREDICT ---
            % Calculate scores: W' * X (add bias row internally or explicitly)
            n_val = size(X_val_fold, 2);
            X_val_bias = [X_val_fold; ones(1, n_val)]; 
            scores = Ws' * X_val_bias;
            [~, y_pred] = max(scores, [], 1);
            
            % --- CALCULATE ACCURACY ---
            % 1. Compute Confusion Matrix (Rows=True, Cols=Pred)
            C = confusionmat(y_val_fold, y_pred);
            % 2. Calculate Accuracy
            fold_accuracy(k, :) = sum(diag(C)) / n_val;

        end
        % Store average performance
        avg_accuracy = mean(fold_accuracy);
        results(i, :) = [mu, avg_accuracy];
        
        fprintf('Mu: %8.5f | Accuracy: %.2f%%\n', mu, avg_accuracy * 100);
    
    end
    
    % 3. Select Winner
    [best_accuracy, idx] = max(results(:, 2));
    best_mu = results(idx, 1);
    
    fprintf('\n----------------------------\n');
    fprintf('BEST MU: %f (Accuracy: %.2f%%)\n', best_mu, best_accuracy * 100);
    fprintf('----------------------------\n');
end