import torch
import torch.nn as nn
import numpy as np
import GenerateData as gd

#Same two step schedule as MATLAB codes but G (equivalent to H in MATLAB) is a GRU (using torch nn.gru class)

# Feature Space Module
class FeatureSpace(nn.Module):
    def __init__(self, feature_type='poly', degree=3, num_centers=10, data_bounds=(-2, 2)):
        super().__init__()
        self.feature_type = feature_type
        
        if feature_type == 'poly':
            self.powers = torch.arange(1, degree + 1, dtype=torch.float32)
            self.dim = degree
        elif feature_type == 'rbf':
            self.centers = torch.linspace(data_bounds[0], data_bounds[1], num_centers)
            self.gamma = 1.0 / ((data_bounds[1] - data_bounds[0]) / num_centers)**2
            self.dim = num_centers
        else:
            raise ValueError("Type must be 'poly' or 'rbf'")

    def forward(self, y):
        if self.feature_type == 'poly':
            features = y ** self.powers.to(y.device)
        else:
            centers = self.centers.to(y.device)
            features = torch.exp(-self.gamma * (y - centers)**2)
            
        features = features - features.mean(dim=0, keepdim=True)
        return features

    def gradients(self, y):
        # Computes analytical dG/dy for the features
        if self.feature_type == 'poly':
            grads = self.powers.to(y.device) * (y ** (self.powers.to(y.device) - 1))
        else:
            centers = self.centers.to(y.device)
            diff = y - centers
            features = torch.exp(-self.gamma * diff**2)
            grads = -2.0 * self.gamma * diff * features
        return grads
        

# RNN Architecture (G_w)
class Gw_RNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=7, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                          num_layers=num_layers, batch_first=True)
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x_past):
        out, _ = self.gru(x_past)
        z = self.readout(out) 
        z = z.squeeze(0)      
        
        z = z - z.mean()
        return z


# Step 1: Outer maximizer 
def step_1_maximizer(model, x_past, y_fixed, feature_module, lr=2e-2, max_steps=300, tol=1e-4, burn_in=0, min_steps=10):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    with torch.no_grad():
        G = feature_module(y_fixed) 
        G_burn = G[burn_in:]        
        eps_reg = 1e-6
        G_cov = G_burn.t() @ G_burn
        G_precision = torch.inverse(G_cov + eps_reg * torch.eye(G_cov.size(0), device=G.device))
        
    prev_loss = float('inf')
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        Z = model(x_past) 
        Z_burn = Z[burn_in:] 
        
        ZtG = Z_burn.t() @ G_burn   
        numerator = ZtG @ G_precision @ ZtG.t() 
        denominator = (Z_burn.t() @ Z_burn) + 1e-8
        
        loss = -0.5 * (numerator.squeeze() / denominator.squeeze())
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        if step >= min_steps and abs(prev_loss - current_loss) < tol:
            print(f"      Step 1 converged at step {step} (Loss: {current_loss:.6f})")
            break
            
        prev_loss = current_loss
        
    with torch.no_grad():
        Z_final = model(x_past)
        
    return model, Z_final.detach()
    
# Step 2: Inner minimizer (via custom gradient descent)
def step_2_minimizer(x, y_init, Z_k, feature_module, sigma_star=0.05, 
                     use_adaptive_lambda=False, lam_c = 2.0, fixed_lambda=500.0, max_iter=20, burn_in=0):
    

    y = y_init.clone().detach()
    T = x.shape[0]
    n_burn = T - burn_in 
    sx_burn = x[burn_in:].std().item()

    if max_iter < 500:
        max_iter = 10000

    eta = 0.1
    Lymin = 1e-5

    # disable autograd for the line search
    with torch.no_grad():
        for js in range(max_iter):
            
            G = feature_module(y) 
            dG_raw = feature_module.gradients(y)

            G_burn = G[burn_in:] 
            dG_burn = dG_raw[burn_in:]

            G_cov = G_burn.t() @ G_burn
            G_precision = torch.inverse(G_cov + 1e-6 * torch.eye(G_burn.shape[1], device=x.device))
            
            Z_burn = Z_k[burn_in:] 
            
            # Vectorized feature projection across all K columns
            ZtG = Z_burn.t() @ G_burn      
            W = G_precision @ ZtG.t()     
            
            # num is the diagonal of ZtG @ W
            num = torch.sum(ZtG.t() * W, dim=0)       
            den = torch.sum(Z_burn**2, dim=0) + 1e-8  
            
            P_l = 0.5 * num / den 
            sigma_l = torch.sqrt(torch.clamp(2.0 * P_l, min=1e-8))
            
            LL = torch.sum(P_l).item()
            
            # Compute analytical gradients
            LLy = torch.zeros_like(y)
            H = torch.zeros_like(y)
            E = Z_burn - (G_burn @ W) 
            
            for k in range(Z_k.shape[1]):
                E_k = E[:, k:k+1] 
                W_k = W[:, k:k+1]
                adj = E_k @ W_k.t()
                
                # Chain Rule through the centering 
                adj_centered = adj - adj.mean(dim=0, keepdim=True)
                dN_dy = 2.0 * torch.sum(adj_centered * dG_burn, dim=1, keepdim=True)
                dP_l_dy = 0.5 * dN_dy / den[k]
                
                LLy[burn_in:] += dP_l_dy
                H[burn_in:] += dP_l_dy * (sigma_star / sigma_l[k].item())
                
            Ly = torch.zeros_like(y)
            Ly[burn_in:] = (y[burn_in:] - x[burn_in:]) / n_burn
            
            if use_adaptive_lambda:
                norm_Ly = torch.norm(Ly).item()
                norm_H = torch.norm(H).item() + 1e-12
                lam = lam_c * max(norm_Ly, sx_burn / np.sqrt(n_burn)) / norm_H
                lam = min(lam, 800.0) 
            else:
                lam = fixed_lambda   
                
            L_old = torch.sum((y[burn_in:] - x[burn_in:])**2).item() / n_burn + lam * LL
            
            total_grad = Ly + lam * LLy
            Lys = torch.sum(total_grad**2).item()
            
            # Line search
            y_new = y - eta * total_grad
            
            G_new = feature_module(y_new)
            G_new_burn = G_new[burn_in:]
            G_cov_new = G_new_burn.t() @ G_new_burn
            G_prec_new = torch.inverse(G_cov_new + 1e-6 * torch.eye(G_new_burn.shape[1], device=x.device))
            

            ZtG_new = Z_burn.t() @ G_new_burn
            num_new = torch.sum((ZtG_new @ G_prec_new) * ZtG_new, dim=1)
            P_l_new = 0.5 * num_new / den
            
            LL_new = torch.sum(P_l_new).item()
            Smax_new = torch.sqrt(torch.clamp(2.0 * P_l_new, min=1e-8)).max().item()
                
            L_new = torch.sum((y_new[burn_in:] - x[burn_in:])**2).item() / n_burn + lam * LL_new
            
            if (L_old - L_new) > 0.5 * eta * Lys:
                eta = 1.01 * eta
            else:
                eta = 0.91 * eta
                
            if eta < 1e-8:
                break
                
            y.data = y_new.data
            
            if (np.sqrt(Lys) / np.sqrt(n_burn) < Lymin):
                break
            
            if not ((Smax_new > sigma_star) or (np.sqrt(Lys) / np.sqrt(n_burn) > Lymin)):
                break

    return y


# Epoch/Stage Loop
def run_epoch_e1(x, x_past, K_e=6, steps_G=300, ymin_iters=20, feature_type='poly', degree=3, tol=2e-4, lam_c=2.0, 
                 sigma_star_accept=0.03, sigma_star_terminate=0.015, max_retries=5, warm_start_G=False, 
                 hidden_dim=5, Ystar=None, burn_in=0, min_steps=10):
    T = x.shape[0]
    
    feature_module = FeatureSpace(feature_type=feature_type, degree=degree, 
                                  data_bounds=(x.min().item(), x.max().item()))
    y_current = x.clone()
    Z_history = []
    

    prev_G_state = None
    
    for k in range(K_e):
        print(f"--- Epoch 1 | Stage {k+1}/{K_e} ---")
        
        valid_Z_found = False
        
        for attempt in range(max_retries):
            #  New fresh model for every attempt
            model_k = Gw_RNN(input_dim=1, hidden_dim=hidden_dim)
            
            # Apply warm start only on the first attempt if enabled and a previous state exists
            if warm_start_G and prev_G_state is not None and attempt == 0:
                model_k.load_state_dict(prev_G_state)
            
            #compiled_model_k = torch.compile(model_k, mode="reduce-overhead")
            compiled_model_k = model_k
            
            compiled_model_k, Z_k = step_1_maximizer(
                model=compiled_model_k, 
                x_past=x_past, 
                y_fixed=y_current, 
                feature_module=feature_module, 
                max_steps=steps_G,
                tol=tol,
                burn_in=burn_in,
                min_steps = min_steps
            )
            
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
            
 
            if max_corr >= sigma_star_accept:
                print(f"      [Attempt {attempt+1}] Accepted extracted correlation: {max_corr:.4f}")
                valid_Z_found = True
                
                # Save the successful model state for the next stage
                prev_G_state = model_k.state_dict()
                break
            elif max_corr <= sigma_star_terminate:
                print(f"      [Attempt {attempt+1}] Correlation {max_corr:.4f} <= {sigma_star_terminate}. True signal is gone.")
                break
            else:
                print(f"      [Attempt {attempt+1}] Weak correlation {max_corr:.4f}. Rejecting and re-rolling RNN...")
                
        if not valid_Z_found:
            print(f"      [!] Terminating stages early. No strong correlation found.")
            break

        Z_history.append(Z_k)
        
        Z_bank = torch.cat(Z_history, dim=1)
        
        # Pass y_current to warm-start the optimization
        y_current = step_2_minimizer(
            x=x, y_init=y_current, Z_k=Z_bank, feature_module=feature_module, 
            use_adaptive_lambda=True, lam_c=lam_c,max_iter=ymin_iters, sigma_star=sigma_star_accept, burn_in=burn_in
        )
        
        mse = torch.mean((x[burn_in:] - y_current[burn_in:])**2).item()
        print(f" Step 2  |  MSE(X, Y): {mse:.4f}")
        
        if Ystar is not None:
            Diff =  ((Ystar[burn_in:] - y_current[burn_in:].detach().numpy().squeeze())**2).mean()
            print(f"   |Y-Y*|^2/|Y*|^2={Diff.item()/(Ystar[burn_in:]**2).mean():.4f}")        
            
    return y_current, Z_history


# Main Execution

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
    

    x_past = torch.cat([torch.zeros(1, 1), x[:-1]], dim=0)
    x_past = x_past.unsqueeze(0) # Shape: (1, T, 1) for the GRU
    

    burn_in_steps = 300
    

    print(f"Equilibrium V = |Pred|^2 = {(P[burn_in_steps:]**2).mean():.4f}")
    
    tol = 1e-4
    sigma_star_accept = 0.03
    sigma_star_terminate = 0.015
    hidden_dim = 2
    min_steps = 4
    ymin_iters = 1500
    lam_c = 3.0
    

    max_retries = 5
    warm_start_G = True 

    
    print(f"Starting OTBP Noise Extraction for GARCH(1,1) (T={T}, burn-in={burn_in_steps})...")
    y_final, Z_bank = run_epoch_e1(
        x=x, 
        x_past=x_past, 
        K_e=10, 
        steps_G=300, 
        ymin_iters=ymin_iters, 
        feature_type='poly', 
        degree=3,
        tol=tol,
        lam_c = lam_c,
        sigma_star_accept=sigma_star_accept,
        sigma_star_terminate=sigma_star_terminate,
        max_retries=max_retries,
        warm_start_G=warm_start_G,
        hidden_dim=hidden_dim,
        Ystar=Ystar,
        burn_in=burn_in_steps,
        min_steps = min_steps
    )
    
    print("\nExtraction Complete.")
    print(f"Equilibrium V = |Pred|^2 = {(P[burn_in_steps:]**2).mean():.4f}")
