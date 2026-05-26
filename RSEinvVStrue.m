load('Oracle_AR2_OTBP_Result.mat');

x_inv_learned = T_inv(y, z_oracle, a, b, k, g_mean, F_mean{1}, lambda, caseF, caseG);
x_inv_oracle = y + z_oracle;

RSE_inv_vs_true_x = sum((x_inv_learned - x).^2)/sum(x.^2);
RSE_inv_vs_oracle_inv = sum((x_inv_learned - x_inv_oracle).^2)/sum(x_inv_oracle.^2);

fprintf('RSE learned inverse vs true x       = %.6f\n', RSE_inv_vs_true_x);
fprintf('RSE learned inverse vs oracle inverse = %.6f\n', RSE_inv_vs_oracle_inv);

figure(102); clf;
scatter(x_inv_oracle, x_inv_learned, 10, 'filled');
grid on;
axis equal;
xlabel('oracle inverse y + z');
ylabel('learned T^{-1}(y,z)');
title('AR(2): learned inverse vs oracle affine inverse');