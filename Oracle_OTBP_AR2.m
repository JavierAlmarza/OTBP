% Oracle_OTBP_AR2.m
% Static OTBP diagnostic for AR(2), using the true conditional mean as z.
%
% Requires:
%   Data_OTBP_with_past.m
%   F_and_Fz.m
%   G_and_Gy.m
%   fg.m
%   P_and_Py_l.m
%   T_inv.m

clearvars; clc;

% ----------------------------
% 1. Generate AR(2) data
% ----------------------------

Example = 'Markov_3';
Data_OTBP_with_past;  % gives x, noise, f_star for Markov_3

% f_star(t) = 0.5*x(t-1)-0.2*x(t-2), already created in your data file.
% Drop first few entries if needed.
valid = 3:length(x);

x = x(valid);
noise = noise(valid);
z_oracle = f_star(valid);

n = length(x);

% ----------------------------
% 2. Hyperparameters
% ----------------------------

caseF = 'L_and_Q';
caseG = 'oneD_L_and_Q';

Kmax = 6;
nsmax = 5000;

eta_0 = 0.01;
eps_fg = 1e-4;
Lymin = eps_fg / sqrt(n);

% Correlation threshold.
sigma_0 = max(0.001, 1.96/sqrt(n));

% Storage
a = cell(Kmax,1);
b = cell(Kmax,1);
f = cell(Kmax,1);
g_mean = cell(Kmax,1);
F_mean = cell(Kmax,1);
lambda = zeros(Kmax,1);
sigma = zeros(Kmax,1);
P = cell(Kmax,1);
Py = cell(Kmax,1);

Cpr = zeros(1,nsmax*Kmax);
Ppr = zeros(1,nsmax*Kmax);
etapr = zeros(1,nsmax*Kmax);
Lampr = zeros(Kmax,nsmax*Kmax);

% ----------------------------
% 3. Fixed oracle F(z)
% ----------------------------

[F_raw, ~] = F_and_Fz(z_oracle, false, caseF);
F_mean_oracle = mean(F_raw,1);
F = F_raw - F_mean_oracle;

% ----------------------------
% 4. Initialize y
% ----------------------------

y = x;
k = 0;
jc = 0;
not_done = true;

fprintf('\n--- Oracle OTBP AR(2) ---\n');
fprintf('n = %d, sigma_0 = %.5f\n', n, sigma_0);

% ----------------------------
% 5. Main stage loop
% ----------------------------

while not_done

    % Current G(y)
    [G_raw, ~] = G_and_Gy(y,false,caseG);
    G_mean = mean(G_raw,1);
    G = G_raw - G_mean;

    % Step 1: find most correlated f(z), g(y)
    switch caseG
        case 'oneD_L'
            my = 1;
        case 'oneD_L_and_Q'
            my = 2;
        otherwise
            error('Unsupported caseG');
    end

    bb0 = rand(my,1) - 0.5;
    [aa,bb] = fg(F,G,eps_fg,bb0);

    ff = F*aa;
    gg = G*bb;
    corr_fg = corr(ff,gg);

    fprintf('\nStage candidate %d: corr = %.6f\n', k+1, corr_fg);

    if abs(corr_fg) < 2*sigma_0
        fprintf('Stopping: corr threshold reached.\n');
        break;
    end

    if k >= Kmax
        fprintf('Stopping: Kmax reached.\n');
        break;
    end

    % Store normalized f
    nf = sqrt(ff'*ff/n);
    k = k + 1;
    a{k} = aa/nf;
    b{k} = bb;
    f{k} = ff/nf;
    F_mean{k} = F_mean_oracle;

    % g_mean for inverse map
    [G_train_raw, ~] = G_and_Gy(y,false,caseG);
    g_mean{k} = mean(G_train_raw*b{k},1);

    % ----------------------------
    % Step 2: update y
    % ----------------------------

    eta = eta_0;
    y_not_converged = true;
    js = 0;

    while y_not_converged && js < nsmax
        js = js + 1;
        jc = jc + 1;

        etapr(jc) = eta;

        [G_raw, Gy] = G_and_Gy(y,true,caseG);
        G_mean = mean(G_raw,1);
        G = G_raw - G_mean;

        Ly = (y - x)/n;
        L_old_cost = sum((y-x).^2)/(2*n);
        LL = 0;

        % Compute penalties and penalty gradients
        for l = 1:k
            [P{l}, Py{l}, b{l}, sigma(l)] = P_and_Py_l(G,Gy,f{l},true);
        end

        % Adaptive lambdas, same structure as current code
        [sigmas_sorted, Isigmas] = sort(sigma(1:k),'ascend');
        sigmas_scaled = min(sigmas_sorted/sigma_0 - 0.8, 30);

        lambda_tmp = zeros(k,1);

        for jj_l = 1:k
            lidx = Isigmas(jj_l);

            denom = sum(sum(Py{lidx}.^2));
            if denom < 1e-14
                c_uv = 0;
            else
                c_uv = sum(sum(Ly.*Py{lidx})) / denom;
            end

            c_uv = min(c_uv,0);
            lambda_tmp(lidx) = max(sigmas_scaled(jj_l) - c_uv, 0);

            Ly = Ly + lambda_tmp(lidx)*Py{lidx};
            LL = LL + lambda_tmp(lidx)*P{lidx};
        end

        lambda(1:k) = lambda_tmp;

        Lampr(1:k,jc) = lambda(1:k);
        Cpr(jc) = L_old_cost;
        Ppr(jc) = LL;

        L_old = L_old_cost + LL;
        Lys = sum(sum(Ly.^2));

        % Gradient step
        y_new = y - eta*Ly;

        % Evaluate new penalized objective
        [G_new_raw, ~] = G_and_Gy(y_new,false,caseG);
        G_new = G_new_raw - mean(G_new_raw,1);

        L_new = sum((y_new-x).^2)/(2*n);
        Smax = -inf;
        LL_new = 0;

        for l = 1:k
            [P_l_new, ~, ~, sigma_l_new] = P_and_Py_l(G_new,0,f{l},false);
            LL_new = LL_new + lambda(l)*P_l_new;
            Smax = max(Smax, sigma_l_new);
        end

        L_new = L_new + LL_new;

        % Step-size adaptation
        if (L_old - L_new > 0.5*eta*Lys - eps_fg)
            eta = 1.01*eta;
            y = y_new;
        else
            eta = 0.81*eta;
        end

        y_not_converged = ...
            (Smax > 0.75*sigma_0 || norm(Ly)/sqrt(n) > (k+1)*Lymin);
    end

    % Update g_mean after final y for this stage
    [G_train_raw, ~] = G_and_Gy(y,false,caseG);
    g_mean{k} = mean(G_train_raw*b{k},1);

    RSE = sum((y-noise).^2)/sum(noise.^2);
    eff_cost = sum((x-y).^2)/(2*n);

    fprintf('End Step 2 stage %d: js=%d, Smax=%.6f, RSE=%.6f, cost=%.6f\n', ...
        k, js, Smax, RSE, eff_cost);
end

fprintf('\nFinal k = %d\n', k);
fprintf('Final RSE y vs true noise = %.6f\n', sum((y-noise).^2)/sum(noise.^2));

% Save useful objects
save('Oracle_AR2_OTBP_Result.mat', ...
    'x','y','noise','z_oracle','a','b','k','g_mean','F_mean','lambda', ...
    'caseF','caseG','sigma_0');