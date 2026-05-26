function [x_out] = T_inv_with_Past(y_sample, x_new, z_0, c, a, b, k, g_mean, F_mean_cell, mu_z_cell, sig_z_cell, lambda, n_train, caseH, caseF, caseG)
% Inverse map simulating the next step distribution for a new sequence of x

[n_samples, ~] = size(y_sample);
x_out = y_sample;

% alpha must use the training sample size
alpha = (n_train - 1) / n_train; 

[G, Gy] = G_and_Gy(y_sample, true, caseG);

% Loop through stages up to k to generate all z^k
for l = 1:k
    x_dummy = [x_new;0];
    z_full_new = Find_z(x_dummy, c{l}, z_0, caseH);
    z_target_raw = z_full_new(end, :);
    
    z_target = (z_target_raw - mu_z_cell{l}) ./ sig_z_cell{l};
    
    [F_raw, ~] = F_and_Fz(z_target, false, caseF); 
    F_s = repmat(F_raw, n_samples, 1);
    F = F_s - F_mean_cell{l}; 
    
    f = F * a{l};
    g = G * b{l};
    gy = Gy * b{l};
    % Formula for x from first order condition of optimal y
    x_out = x_out + lambda(l) * gy .* (f - (g - g_mean{l}));
end
end