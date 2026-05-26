% Offline conditional generation

N_samples = 5000; 

% Provide new out of sample x sequence here 
Data_OTBP_with_past
x_new = x;

% Sample N shocks from the previously obtained barycentric y
idx = randi(size(y, 1), N_samples, 1);
y_sample = y(idx, :);

% Compute g_mean from the obtained barycenter
[G_train, ~] = G_and_Gy(y, false, caseG);
g_mean = cell(k, 1);
for l = 1:k
    g_mean{l} = mean(G_train * b{l}, 1);
end

% Push the N samples through the offline inverse map (and push x_new to generate new z)
n_train = size(y, 1);
k_override = 1;
x_generated = T_inv_with_Past(y_sample, x_new, z_0, c, a, b, k, g_mean, F_mean_cell, mu_z_cell, sig_z_cell, lambda, n_train, caseH, caseF, caseG);
%x_generated = T_inv_with_Past(y_sample, x_new, z_0, c, a, b, k_override, g_mean, F_mean_cell, mu_z_cell, sig_z_cell, lambda, n_train, caseH, caseF, caseG);


% Compute true distribution of next step from synthetic data
true_mean = 0;
true_sigma = 1;

if strcmp(Example, 'Garch')
    omega = 0.05; alpha = 0.15; beta = 0.80;
    next_var_ev = omega + alpha * x_new(end)^2 + beta * var_ev(end);
    true_sigma = sqrt(next_var_ev);
    true_mean = 0;
    
elseif strcmp(Example, 'Markov_3')
    true_mean = 0.5 * x_new(end) - 0.2 * x_new(end-1);
    true_sigma = 0.4;

elseif strcmp(Example, 'Markov_1')
    true_mean = 0.5 * x_new(end) ;
    true_sigma = 1.5;


elseif strcmp(Example, 'Kalman')
    phi = 0.8; sig_w = 0.4; sig_v = 0.4;
    P_pred = sig_w^2 / (1 - phi^2); 
    for j = 1:500 % Burn-in to steady state K
        K = P_pred / (P_pred + sig_v^2);
        P_upd = (1 - K) * P_pred;
        P_pred = phi^2 * P_upd + sig_w^2;
    end
    K_steady = P_pred / (P_pred + sig_v^2);
    Innovation_var = P_pred + sig_v^2;
    
    % Update the final state and push it forward one step
    s_upd = P_kalman(end) + K_steady * epsi(end);
    true_mean = phi * s_upd;
    true_sigma = sqrt(Innovation_var);
end


% Print diagnostics
[~, dim_x] = size(x_generated);

if dim_x == 1
    x_gen_sorted = sort(x_generated);
    
    % 1-Wasserstein metric is L1 distance of inverse cdfs
    p_quantiles = ((1:N_samples)' - 0.5) / N_samples;
    x_true_quantiles = norminv(p_quantiles, true_mean, true_sigma);
    W1_dist = mean(abs(x_gen_sorted - x_true_quantiles));
    
    % Kolmogorov-Smirnov 
    cdf_true = normcdf(x_gen_sorted, true_mean, true_sigma);
    cdf_emp = (1:N_samples)' / N_samples;
    KS_stat = max(abs(cdf_emp - cdf_true));
    
    fprintf('\n--- Predictive Density Fit (1D) ---\n');
    fprintf('True Conditional Mean        : %.4f\n', true_mean);
    fprintf('True Conditional Vol (sigma) : %.4f\n', true_sigma);
    fprintf('1-Wasserstein Distance (W1)  : %.4f\n', W1_dist);
    fprintf('Kolmogorov-Smirnov Stat (KS) : %.4f\n', KS_stat);
else
    fprintf('\n--- Predictive Density Fit ---\n');
    fprintf('Dimension dim_x = %d detected. Exact 1D W1/KS bypassed.\n', dim_x);
    % TODO: implement n-dimensional metrics (eg. MMD)
end


figure(7)
clf


histogram(x_generated, 50, 'Normalization', 'pdf', 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'w', 'FaceAlpha', 0.75);
hold on;


x_true_sample = true_mean + true_sigma * randn(N_samples, dim_x);
histogram(x_true_sample, 50, 'Normalization', 'pdf', 'FaceColor', [0.8 0.3 0.2], 'EdgeColor', 'w', 'FaceAlpha', 0.5);


if dim_x == 1
    x_grid = linspace(min(min(x_generated), min(x_true_sample)), max(max(x_generated), max(x_true_sample)), 300);
    pdf_true = normpdf(x_grid, true_mean, true_sigma);
    plot(x_grid, pdf_true, 'k--', 'LineWidth', 2);
    legend('T^{-1} Generated Density', 'True Sampled Density', 'True Analytical PDF', 'Location', 'best');
else
    legend('T^{-1} Generated Density', 'True Sampled Density', 'Location', 'best');
end

title(sprintf('Offline Predictive Density (N = %d)', N_samples));
xlabel('Generated x');
ylabel('Density');
hold off;
