import torch
import torch.nn as nn
import numpy as np
import GenerateData as gd

#Same schedule as FinalMaxMin but G is a simple recursion following the GARCH model

# ==========================================
# 1. Feature Space Module
# ==========================================
class FeatureSpace(nn.Module):
    def __init__(self, feature_type='poly', degree=3, num_centers=10, data_bounds=(-2, 2)):
        super().__init__()
        self.feature_type = feature_type
        
        if feature_type == 'poly':
            # Polynomial powers [1, 2, ..., degree] (omitting 0)
            self.powers = torch.arange(1, degree + 1, dtype=torch.float32)
            self.dim = degree
        elif feature_type == 'rbf':
            # Fixed centers spread across the expected data range
            self.centers = torch.linspace(data_bounds[0], data_bounds[1], num_centers)
            # Bandwidth heuristic
            self.gamma = 1.0 / ((data_bounds[1] - data_bounds[0]) / num_centers)**2
            self.dim = num_centers
        else:
            raise ValueError("Type must be 'poly' or 'rbf'")

    def forward(self, y):
        # y shape: (T, 1)
        if self.feature_type == 'poly':
            features = y ** self.powers.to(y.device)
        else:
            centers = self.centers.to(y.device)
            features = torch.exp(-self.gamma * (y - centers)**2)
            
        # Strict centering equivalent to G = G - mean(G, 1)
        features = features - features.mean(dim=0, keepdim=True)
        return features



# ==========================================
# 2. Parametric GARCH Module (Softmax)
# ==========================================
class Gw_GARCH(nn.Module):
    def __init__(self):
        super().__init__()
        # Completely random initialization (no hints)
        self.raw_a = nn.Parameter(torch.randn(1)) 
        
        # Softmax Simplex: [b, c, dummy] randomly initialized
        self.raw_simplex = nn.Parameter(torch.randn(3))

    def forward(self, x_past):
        # x_past shape: (1, T, 1), representing x_{t-1}
        x_sq = x_past[0, :, 0] ** 2 
        T = x_sq.shape[0]
        
        # 1. Apply Parameter Constraints
        a = torch.exp(self.raw_a) + 1e-6 # Strictly > 0
        
        # Softmax perfectly guarantees b > 0, c > 0, and b + c < 1
        probs = torch.softmax(self.raw_simplex, dim=0)
        b = probs[0] * 0.999
        c = probs[1] * 0.999
        
        # 2. Calculate Initial Unconditional State
        z_0 = a / (1.0 - b - c)
        
        # 3. Unroll the GARCH recurrence
        z_t = z_0
        z_list = []
        for t in range(T):
            z_t = a + b * x_sq[t] + c * z_t
            z_list.append(z_t)
            
        # Strictly enforce (T, 1) shape to prevent 3D transpose crash
        z = torch.stack(z_list).view(-1, 1) 
        
        # 4. Feature Space: f(z) = sqrt(z)
        f_z = torch.sqrt(z)
        
        # Strictly center the feature
        f_z = f_z - f_z.mean()
        return f_z
        
# ==========================================
# 3. Step 1: Maximizer (Update G_w)
# ==========================================
def step_1_maximizer(model, x_past, y_fixed, feature_module, lr=2e-2, max_steps=300, tol=1e-4, burn_in=0, min_steps=10):
    """
    Optimizes the parametric GARCH G_w with early stopping based on a tolerance threshold.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    with torch.no_grad():
        G = feature_module(y_fixed) # Shape: (T, D_y)
        G_burn = G[burn_in:]        # Apply burn-in slice
        eps_reg = 1e-6
        G_cov = G_burn.t() @ G_burn
        # Tikhonov regularization
        G_precision = torch.inverse(G_cov + eps_reg * torch.eye(G_cov.size(0), device=G.device))
        
    prev_loss = float('inf')
    
    for step in range(max_steps):
        
        optimizer.zero_grad()
        
        Z = model(x_past) # Shape: (T, 1)
        Z_burn = Z[burn_in:] # Apply burn-in slice
        
        ZtG = Z_burn.t() @ G_burn   # Shape: (1, D_y)
        numerator = ZtG @ G_precision @ ZtG.t() 
        denominator = (Z_burn.t() @ Z_burn) + 1e-8
        
        # Maximize Rayleigh quotient by minimizing its negative
        loss = -0.5 * (numerator.squeeze() / denominator.squeeze())
        
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        # --- THE FIX: Only allow early stopping AFTER min_steps ---
        if step >= min_steps and abs(prev_loss - current_loss) < tol:
            print(f"      Step 1 converged at step {step} (Loss: {current_loss:.6f})")
            break
            
        prev_loss = current_loss
        
    return model, Z.detach()

# ==========================================
# 4. Step 2: Minimizer (Update Y via Custom GD)
# ==========================================
def step_2_minimizer(x, y_init, Z_k, feature_module, sigma_star=0.015, 
                     use_adaptive_lambda=False, lam_c=2.0, fixed_lambda=500.0, max_iter=20, burn_in=0):
    
    # 1. Initialize y from the PREVIOUS stage's output, not from x!
    y = y_init.clone().detach().requires_grad_(True)
    T = x.shape[0]
    n_burn = T - burn_in # Adjust n for the baseline and MSE scaling
    sx_burn = x[burn_in:].std().item()

    # The custom GD line search requires more steps than L-BFGS.
    # Matching the override logic from the GRU script
    if max_iter < 500:
        max_iter = 10000

    eta = 0.1
    Lymin = 1e-5

    # ==========================================
    # CUSTOM GRADIENT DESCENT (Replacing L-BFGS)
    # ==========================================
    for js in range(max_iter):
        
        G = feature_module(y) 
        G_burn = G[burn_in:] # Apply burn-in slice
        G_cov = G_burn.t() @ G_burn
        G_precision = torch.inverse(G_cov + 1e-6 * torch.eye(G_burn.shape[1], device=x.device))
        
        LL = 0.0
        H = torch.zeros_like(y)
        LLy = torch.zeros_like(y)
        Smax = -1.0
        
        # Iterate over each extracted 1D dependency from the past stages
        for k in range(Z_k.shape[1]):
            Z_current = Z_k[:, k:k+1] # Shape: (T, 1)
            Z_current_burn = Z_current[burn_in:] # Apply burn-in slice
            
            ZtG = Z_current_burn.t() @ G_burn   # Shape: (1, D_y)
            numerator = ZtG @ G_precision @ ZtG.t() 
            denominator = (Z_current_burn.t() @ Z_current_burn) + 1e-8
            
            # The exact analytical solution for max_g Pen(g(Y), Z^k)
            P_l = 0.5 * (numerator.squeeze() / denominator.squeeze())
            LL = LL + P_l
            
            # Reconstruct correlation sigma from penalty
            sigma_l = torch.sqrt(torch.clamp(2.0 * P_l, min=1e-8))
            Smax = max(Smax, sigma_l.item())
            
            # Compute analytical gradients using Autograd
            is_last = (k == Z_k.shape[1] - 1)
            Py_l = torch.autograd.grad(P_l, y, retain_graph=not is_last)[0]
            
            LLy = LLy + Py_l
            # The exact MATLAB math trick: Py{l}*(sigma_0/sigma{l})
            H = H + Py_l * (sigma_star / sigma_l.item())
            
        Ly = torch.zeros_like(y)
        # Compute MSE gradient only on the active slice
        Ly[burn_in:] = (y[burn_in:] - x[burn_in:]) / n_burn
        
        if use_adaptive_lambda:
            norm_Ly = torch.norm(Ly).item()
            norm_H = torch.norm(H).item() + 1e-12
            
            # MATLAB logic for lambda utilizing lam_c
            lam = lam_c * max(norm_Ly, sx_burn / np.sqrt(n_burn)) / norm_H
            lam = min(lam, 800.0) # Keeping your safety clamp
        else:
            lam = fixed_lambda
            
        L_old = torch.sum((y[burn_in:] - x[burn_in:])**2).item() / n_burn + lam * LL.item()
        
        total_grad = Ly + lam * LLy
        Lys = torch.sum(total_grad**2).item()
        
        # Gradient Descent Step
        with torch.no_grad():
            y_new = y - eta * total_grad
            
            # Evaluate L_new and Smax for the line search and convergence criteria
            G_new = feature_module(y_new)
            G_new_burn = G_new[burn_in:]
            G_cov_new = G_new_burn.t() @ G_new_burn
            G_prec_new = torch.inverse(G_cov_new + 1e-6 * torch.eye(G_new_burn.shape[1], device=x.device))
            
            LL_new = 0.0
            Smax_new = -1.0
            for k in range(Z_k.shape[1]):
                Z_current = Z_k[:, k:k+1]
                Z_current_burn = Z_current[burn_in:]
                ZtG = Z_current_burn.t() @ G_new_burn
                num = ZtG @ G_prec_new @ ZtG.t()
                den = (Z_current_burn.t() @ Z_current_burn) + 1e-8
                P_l_new = 0.5 * (num.squeeze() / den.squeeze()).item()
                LL_new += P_l_new
                Smax_new = max(Smax_new, np.sqrt(max(2.0 * P_l_new, 0.0)))
                
            L_new = torch.sum((y_new[burn_in:] - x[burn_in:])**2).item() / n_burn + lam * LL_new
            
        # Line search eta update
        if (L_old - L_new) > 0.5 * eta * Lys:
            eta = 1.01 * eta
        else:
            eta = 0.91 * eta
            
        # --- SAFEGUARD 1: ETA COLLAPSE ---
        if eta < 1e-8:
            break
            
        y.data = y_new.data
        
        # --- SAFEGUARD 2: VANISHING GRADIENT OVERRIDE ---
        if (np.sqrt(Lys) / np.sqrt(n_burn) < Lymin):
            break
        
        # Same termination criterion as OTBP.m
        y_not_converged = (Smax_new > sigma_star) or (np.sqrt(Lys) / np.sqrt(n_burn) > Lymin)
        if not y_not_converged:
            break

    return y.detach()

# ==========================================
# 5. The Epoch Loop
# ==========================================
def run_epoch_e1(x, x_past, K_e=6, steps_G=300, ymin_iters=20, feature_type='poly', degree=3, 
                 Ystar=None, burn_in=0, tol=2e-4, lam_c=2.0, sigma_star=0.015, min_steps=10):
    T = x.shape[0]
    
    feature_module = FeatureSpace(feature_type=feature_type, degree=degree, 
                                  data_bounds=(x.min().item(), x.max().item()))
    y_current = x.clone()
    Z_history = []
    
    for k in range(K_e):
        print(f"--- Epoch 1 | Stage {k+1}/{K_e} ---")
        
        # Instantiate the Parametric GARCH model
        model_k = Gw_GARCH()
        
        # No need for torch.compile on a 3-parameter model, eager executes instantly
        model_k, Z_k = step_1_maximizer(
            model=model_k, 
            x_past=x_past, 
            y_fixed=y_current, 
            feature_module=feature_module, 
            max_steps=steps_G,
            tol=tol,
            burn_in=burn_in,
            min_steps=min_steps
        )
        
        # Dynamic termination
        with torch.no_grad():
            G = feature_module(y_current)
            G_burn = G[burn_in:]
            G_cov = G_burn.t() @ G_burn
            G_precision = torch.inverse(G_cov + 1e-6 * torch.eye(G_burn.shape[1], device=x.device))
            Z_k_burn = Z_k[burn_in:]
            ZtG = Z_k_burn.t() @ G_burn
            numerator = ZtG @ G_precision @ ZtG.t() 
            denominator = (Z_k_burn.t() @ Z_k_burn) + 1e-8
            final_loss = -0.5 * (numerator.squeeze() / denominator.squeeze()).item()
            
        max_corr = np.sqrt(max(-2.0 * final_loss, 0.0))
        print(f"      Maximal extracted correlation: {max_corr:.4f}")
        
        # Print the fitted parameters for interpretability
        with torch.no_grad():
            a = torch.exp(model_k.raw_a).item()
            probs = torch.softmax(model_k.raw_simplex, dim=0)
            b = probs[0].item() * 0.999
            c = probs[1].item() * 0.999
            print(f"      Fitted GARCH params -> a: {a:.4f}, b: {b:.4f}, c: {c:.4f}")        
            
        if max_corr < sigma_star:
            print(f"      [!] Correlation {max_corr:.4f} < {sigma_star}. Terminating stages early.")
            break
        # --------------------------------
            
        Z_history.append(Z_k)
        
        Z_bank = torch.cat(Z_history, dim=1)
        
        # Pass y_current as y_init to warm-start the optimization
        y_current = step_2_minimizer(
            x=x, y_init=y_current, Z_k=Z_bank, feature_module=feature_module, 
            use_adaptive_lambda=True, lam_c=lam_c, max_iter=ymin_iters, 
            sigma_star=sigma_star, burn_in=burn_in
        )
        
        # Compute final MSE ignoring the burn_in
        mse = torch.mean((x[burn_in:] - y_current[burn_in:])**2).item()
        print(f" Step 2  |  MSE(X, Y): {mse:.4f}")
        if Ystar is not None:
            Diff =  ((Ystar[burn_in:] - y_current[burn_in:].detach().numpy().squeeze())**2).mean()
            print(f"   |Y-Y*|^2/|Y*|^2={Diff.item()/(Ystar[burn_in:]**2).mean():.4f}")        
            
    return y_current, Z_history

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    
    T = 3000
    
    mode_data = "GARCH"
    print(" ")
    print(f"Input data: {mode_data}")
    
    if mode_data == "Kalman":
        x_np, Ztrue = gd.generate_kalman_data(T, 0.8, 1, 0.4, 0.4)
        x_np = x_np.reshape(-1)
        P = gd.kalman_predictors(x_np, 0.8, 1, 0.4, 0.4)
        Ystar = x_np - P
    elif mode_data == "GARCH":
        omega, alpha, beta = 0.05, 0.18, 0.75
        x_np, Ztrue, epsi = gd.generate_garch_data(T,omega,alpha,beta)
        x_np = x_np.reshape(-1)
        P = np.sqrt(Ztrue) - np.sqrt(Ztrue).mean()
        Ystar = epsi * (np.sqrt(Ztrue).mean())
    elif mode_data == "AR":
        phi_coeffs = [0.5, -0.2, 0.1]
        noise_variance = 1
        x_np, Ztrue = gd.generate_ar_data(phi_coeffs, sigma2=noise_variance, n_steps=T)
        P = Ztrue
        Ystar = x_np - P
    else:
        raise ValueError(f"Unknown data mode: {mode_data}")
    
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(1) # Shape: (T, 1)
    
    # 2. Construct x_past respecting the filtration (shift by 1)
    x_past = torch.cat([torch.zeros(1, 1), x[:-1]], dim=0)
    x_past = x_past.unsqueeze(0) # Shape: (1, T, 1) 
    
    # ------------------------------------------
    # HYPERPARAMETER CONTROL CENTER
    # ------------------------------------------
    burn_in_steps = 300
    tol = 2e-4
    sigma_star = 0.015
    min_steps = 4
    ymin_iters = 2000
    lam_c = 3.0
    # ------------------------------------------
    
    # Update expected Equilibrium output to also ignore the burn-in period
    print(f"Equilibrium V = |Pred|^2 = {(P[burn_in_steps:]**2).mean():.4f}")

    # 3. Run the algorithm
    print(f"Starting Parametric OTBP Noise Extraction for GARCH(1,1) (T={T}, burn-in={burn_in_steps})...")
    y_final, Z_bank = run_epoch_e1(
        x=x, 
        x_past=x_past, 
        K_e=10, 
        steps_G=300, 
        ymin_iters=ymin_iters, 
        feature_type='poly', 
        degree=3,
        Ystar=Ystar,
        burn_in=burn_in_steps,
        tol=tol,
        lam_c=lam_c,
        sigma_star=sigma_star,
        min_steps=min_steps
    )
    
    print("\nExtraction Complete.")
    print(f"Equilibrium V = |Pred|^2 = {(P[burn_in_steps:]**2).mean():.4f}")