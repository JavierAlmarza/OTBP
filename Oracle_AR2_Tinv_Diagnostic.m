% Oracle_AR2_Tinv_Diagnostic.m
% Tests conditional simulation using oracle AR(2) state.

clearvars; clc;

load('Oracle_AR2_OTBP_Result.mat');

N_samples = 5000;

% Choose a new observed AR(2) history.
% For example use the last two values from the training sample.
x_tm1 = x(end);
x_tm2 = x(end-1);

true_mean = 0.5*x_tm1 - 0.2*x_tm2;
true_sigma = 1.5;

z_new = true_mean;  % oracle scalar predictive state

% Sample y's from learned barycenter
idx = randi(length(y), N_samples, 1);
y_sample = y(idx);

% Evaluate inverse map at oracle z_new
z_eval = repmat(z_new, N_samples, 1);

x_gen = T_inv(y_sample, z_eval, a, b, k, g_mean, F_mean{1}, lambda, caseF, caseG);

% True conditional sample
x_true = true_mean + true_sigma*randn(N_samples,1);

% 1D W1 distance
xg = sort(x_gen);
xt = sort(x_true);
W1_emp = mean(abs(xg - xt));

% Against analytic quantiles
p = ((1:N_samples)' - 0.5)/N_samples;
x_true_q = true_mean + true_sigma*norminv(p);
W1_analytic = mean(abs(xg - x_true_q));

KS_stat = max(abs((1:N_samples)'/N_samples - normcdf(xg,true_mean,true_sigma)));

fprintf('\n--- Oracle AR(2) conditional inverse diagnostic ---\n');
fprintf('true_mean     = %.6f\n', true_mean);
fprintf('true_sigma    = %.6f\n', true_sigma);
fprintf('mean(x_gen)   = %.6f\n', mean(x_gen));
fprintf('std(x_gen)    = %.6f\n', std(x_gen));
fprintf('W1 empirical  = %.6f\n', W1_emp);
fprintf('W1 analytic   = %.6f\n', W1_analytic);
fprintf('KS analytic   = %.6f\n', KS_stat);

figure(101); clf;
histogram(x_gen, 50, 'Normalization','pdf');
hold on;
histogram(x_true, 50, 'Normalization','pdf');
grid on;
legend('T^{-1} generated','True conditional sample');
title('Oracle AR(2): conditional inverse diagnostic');
xlabel('x');
ylabel('density');