% Define label:
% glioma = 1 (Train: 1321 samples, Test: 300 samples)
% meningioma = 2 (Train: 1339 samples, Test: 306 samples)
% notumor = 3 (Train: 1595 samples, Test: 405 samples)
% pituitary = 4 (Train: 1457 samples, Test: 300 samples)

% Construct labels for training and test sets
ytr = [ones(1, 1321) ones(1, 1339)*2 ones(1, 1595)*3 ones(1, 1457)*4];
yte = [ones(1, 300) ones(1, 306)*2 ones(1, 405)*3 ones(1, 300)*4];

% Set hyperparameters (d and B)
d = 8;
B = 9;

% Prepare the second pair of data matrices for HOG of the input data
n_tr = size(Xtr, 2);
n_features = ((128 - d) / (d / 2) + 1)^2 * B;
H = zeros(n_features, n_tr); % Initialize pre-allocated space to enhance speed
for i = 1:n_tr
    xi = Xtr(:,i);
    mi = reshape(xi,128,128);
    hi = hog20(mi,d,B);
    H(:, i) = hi;
end

n_te = size(Xte, 2);
Hte = zeros(n_features, n_te); % Initialize pre-allocated space to enhance speed
for i = 1:n_te
    xi = Xte(:,i);
    mi = reshape(xi,128,128);
    hi = hog20(mi,d,B);
    Hte(:, i) = hi;
end
%% Z-Score Normalization:
% 1. Calculate Mean and Standard Deviation of the Training 
ori_mean = mean(Xtr, 2);
ori_std = std(Xtr, 0, 2) + 1e-6; % Add tiny epsilon to prevent division by zero
hog_mean = mean(H, 2);
hog_std = std(H, 0, 2) + 1e-6; % Add tiny epsilon to prevent division by zero

% 2. Normalize Training Data (Z-Score)
% (Value - Mean) / Std
Xtr_norm = bsxfun(@rdivide, bsxfun(@minus, Xtr, ori_mean), ori_std);
H_norm = bsxfun(@rdivide, bsxfun(@minus, H, hog_mean), hog_std);

% 3. Normalize Test Data using TRAINING statistics
% Important: Do NOT recalculate mean/std on Hte. Use the Training ones.
Xte_norm = bsxfun(@rdivide, bsxfun(@minus, Xte, ori_mean), ori_std);
Hte_norm = bsxfun(@rdivide, bsxfun(@minus, Hte, hog_mean), hog_std);

% 4. Create full, normalized data sets
Dtr_norm = [Xtr_norm; ytr];
Dhtr_norm = [H_norm; ytr];
Dte_norm = [Xte_norm; yte];
Dhte_norm = [Hte_norm; yte];

%% Hyperparameter Tuning for mu_grid:
% Selecting the mu that results in the best mean accuracy in 5-fold cross
% validation
K_folds = 5;
K = 4;
mu_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1];
iter = 500;
%%
disp('------------- RAW FEATURES -------------')
disp('Starting Grid Search for Best Mean Accuracy (Original Features)...');
oriStart = tic; % Start Timer
oriCVStart = tic;

[best_mu_ori, best_accuracy_ori] = ...
        tune_hyperparameter(Xtr_norm, ytr, mu_grid, K_folds, K, iter);

% Second round of mu selection:
disp('------- Second round of Mu Selection for Raw Data -------')
mu_grid_2 = [best_mu_ori*0.5, best_mu_ori*2, best_mu_ori*3, best_mu_ori*5];
[best_mu_ori_2, best_accuracy_ori_2] = ...
        tune_hyperparameter(Xtr_norm, ytr, mu_grid_2, K_folds, K, iter);

if best_accuracy_ori_2 > best_accuracy_ori
    best_mu_ori_final = best_mu_ori_2;
else
    best_mu_ori_final = best_mu_ori;
end
oriCVEnd = toc(oriCVStart);
fprintf('Final Selected mu (Raw Data)%.5f%', best_mu_ori_final);

%%
% Train the entire training set on best mu
oriTrainStart = tic;
[Ws, ~] = SRMCC_bfgsML(Dtr_norm, 'f_SRMCC', 'g_SRMCC', best_mu_ori_final, K, iter); % Train on best mu
oriTrainEnd = toc(oriTrainStart);

% Make prediction:
X_test_bias = [Xte_norm; ones(1, n_te)]; 
scores = Ws' * X_test_bias;
[~, y_pred] = max(scores, [], 1);

oriEnd = toc(oriStart);

% Performance evaluation for training
% 1. Compute Confusion Matrix (Rows=Pred, Cols=True)
C_ori_test = confusionmat(yte, y_pred);

% 2. Calculate Accuracy
accuracy_ori_test = sum(diag(C_ori_test)) / n_te;

fprintf('Raw Features Test Accuracy: %.2f%%\n', accuracy_ori_test * 100);
fprintf('Cross Validation Speed Performance: %f\n', oriCVEnd);
fprintf('Training Speed Performance: %f\n', oriTrainEnd);
fprintf('Total Speed Performance (CV + Training + Prediction): %f\n', oriEnd);

%% For HOG
disp('------------- HOG FEATURES -------------')
disp('Starting Grid Search for Best Mean Accuracy (HOG Features)...');

hogStart = tic; % Start Timer
hogCVStart = tic;
[best_mu_hog, best_accuracy_hog] = ...
        tune_hyperparameter(H_norm, ytr, mu_grid, K_folds, K, iter);

% Second round of mu selection:
disp('------- Second round of Mu Selection -------')
mu_grid_2 = [best_mu_hog*0.5, best_mu_hog*2, best_mu_hog*3, best_mu_hog*5];

if best_accuracy_hog_2 > best_accuracy_hog
    best_mu_hog_final = best_mu_hog_2;
else
    best_mu_hog_final = best_mu_hog;
end
hogCVEnd = toc(hogCVStart);
fprintf('Final Selected mu %.5f%', best_mu_hog_final);

%%
% Train the entire HOG training set on best mu
hogTrainStart = tic;
[Ws_hog, ~] = SRMCC_bfgsML(Dhtr_norm, 'f_SRMCC', 'g_SRMCC', best_mu_hog_final, K, iter); % Train on best mu
hogTrainEnd = toc(hogTrainStart);
% Make prediction:
X_test_bias_hog = [Hte_norm; ones(1, n_te)]; 
scores_hog = Ws_hog' * X_test_bias_hog;
[~, y_pred_hog] = max(scores_hog, [], 1);

hogEnd = toc(hogStart); % End Timer
%%
% Performance evaluation for training
% 1. Compute Confusion Matrix (Rows=True, Cols=Pred)
C_hog_test = confusionmat(y_pred_hog, yte);
%%
% 2. Calculate Accuracy
accuracy_hog_test = sum(diag(C_hog_test)) / n_te;

fprintf('HOG Test Accuracy: %.2f%%\n', accuracy_hog_test * 100);

fprintf('Cross Validation Speed Performance: %f\n', hogCVEnd);
fprintf('Training Speed Performance: %f\n', hogTrainEnd);
fprintf('Total Speed Performance (CV + Training + Prediction): %f\n', hogEnd);

%% Plotting
% HOG Features Data
% [Round 1 (Broad) + Round 2 (Refined)]
mu_hog = [0.0001, 0.0010, 0.0100, 0.1000, 1.0000, 0.0500, 0.2000, 0.3000, 0.5000];
acc_hog = [87.04, 87.29, 87.39, 87.53, 86.27, 87.76, 87.10, 87.48, 87.15];

% Raw Features Data
% [Round 1 (Broad) + Round 2 (Refined)]
mu_raw = [0.0001, 0.0010, 0.0100, 0.1000, 1.0000, 0.0500, 0.2000, 0.3000, 0.5000];
acc_raw = [79.64, 79.94, 80.22, 81.18, 80.44, 81.15, 81.27, 81.22, 81.09];

% --- Sort Data for Smooth Plotting ---
[mu_hog_sorted, idx_hog] = sort(mu_hog);
acc_hog_sorted = acc_hog(idx_hog);

[mu_raw_sorted, idx_raw] = sort(mu_raw);
acc_raw_sorted = acc_raw(idx_raw);

% --- Plotting ---
figure('Color', 'w'); % White background
hold on;

% 1. Plot HOG Features (Blue Line with Circles)
p1 = semilogx(mu_hog_sorted, acc_hog_sorted, '-bo', ...
    'LineWidth', 2, 'MarkerFaceColor', 'b', 'MarkerSize', 6, ...
    'DisplayName', 'HOG Features');

% 2. Plot Raw Features (Red Line with Squares)
p2 = semilogx(mu_raw_sorted, acc_raw_sorted, '-rs', ...
    'LineWidth', 2, 'MarkerFaceColor', 'r', 'MarkerSize', 6, ...
    'DisplayName', 'Raw Features');

% 3. Highlight the Best Points for Each
[max_acc_hog, idx_best_hog] = max(acc_hog_sorted);
best_mu_hog = mu_hog_sorted(idx_best_hog);

[max_acc_raw, idx_best_raw] = max(acc_raw_sorted);
best_mu_raw = mu_raw_sorted(idx_best_raw);

plot(best_mu_hog, max_acc_hog, 'bp', 'MarkerSize', 14, ...
    'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');
plot(best_mu_raw, max_acc_raw, 'rp', 'MarkerSize', 14, ...
    'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'HandleVisibility', 'off');

% --- Annotations ---
% Add text labels pointing to the best values
text(best_mu_hog, max_acc_hog + 0.5, sprintf('Best HOG: %.2f%%', max_acc_hog), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'b');
text(best_mu_raw, max_acc_raw - 0.5, sprintf('Best Raw: %.2f%%', max_acc_raw), ...
    'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'Color', 'r');

% --- Formatting ---
grid on;
legend([p1, p2], 'Location', 'best');
xlabel('Regularization Parameter \mu (Log Scale)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Validation Accuracy (%)', 'FontSize', 12, 'FontWeight', 'bold');
title('Performance Comparison: HOG vs. Raw Features', 'FontSize', 14);

% Adjust axis limits for clarity
xlim([0.00008, 1.2]);
ylim([78, 90]); % Covers range of both datasets

hold off;