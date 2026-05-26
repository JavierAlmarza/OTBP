
% --- Diagnostic 2: FOC Transfer Function Plot ---
y_grid = linspace(-4, 4, 1000)';
x_mapped = y_grid;
[G_grid, Gy_grid] = G_and_Gy(y_grid, true, caseG);

for l = 1:k
    % Re-evaluate the target z for the specific stage
    x_dummy = [x_new; 0];
    z_full_diag = Find_z(x_dummy, c{l}, z_0, caseH);
    z_target_diag = z_full_diag(end, :);
    z_target_norm = (z_target_diag - mu_z_cell{l}) ./ sig_z_cell{l};
    
    [F_raw, ~] = F_and_Fz(z_target_norm, false, caseF);
    F_s = repmat(F_raw, 1000, 1);
    F_grid = F_s - F_mean_cell{l};
    
    f_grid = F_grid * a{l};
    g_grid = G_grid * b{l};
    gy_grid = Gy_grid * b{l};
    
    x_mapped = x_mapped + lambda(l) * gy_grid .* (f_grid - (g_grid - g_mean{l}));
end

figure(10);
yyaxis left;
plot(y_grid, x_mapped, 'b-', 'LineWidth', 2); hold on;
plot(y_grid, y_grid, 'k--', 'LineWidth', 1); % Identity baseline
ylabel('Mapped Output x = T^{-1}(y)');

yyaxis right;
plot(y_grid, normpdf(y_grid, 0, 1), 'r-', 'LineWidth', 1.5, 'Color', [0.8 0.3 0.2 0.4]);
ylabel('Density of Input Noise y \sim N(0,1)');
ylim([0, 0.5]);

xlabel('Input Noise y');
title('FOC Transfer Function (Fixed z_{T+1})');
grid on; hold off;