n=3100; % number of observations.

switch Example
    case 'Markov_1'
        n_burn=300;
        n=n+n_burn;
        x=zeros(n,1);
        f_star=zeros(n,1);
        x(1)=0;
        noise=1.5*randn(n,1);
        for j=2:n
            f_star(j)=0.5*x(j-1);
            x(j)=0.5*x(j-1)+noise(j);
        end
        x = x(n_burn+1:end);
        f_star = f_star(n_burn+1:end);
        noise = noise(n_burn+1:end);


    case 'Markov_2' % I think Arch? Not quite, and not Markov
        x=zeros(n,1);
        x(1)=0;
        noise=1.*randn(n,1);
        for j=2:n
            x(j)=0*0.5*x(j-1)+sqrt((1+0.2*(noise(j-1)^2)))*noise(j);
        end
    case 'Markov_3'
        n_burn = 300;
        n = n+n_burn;
        x=zeros(n,1);
        x(1)=0; x(2)=0;
        f_star=zeros(n,1);
        f_star(1)=0; f_star(2)=0;
        noise=0.4*randn(n,1);
        for j=3:n
            f_star(j)=0.5*x(j-1)-0.2*x(j-2);
            x(j)=f_star(j)+noise(j);
        end
        x = x(n_burn+1:end);
        f_star = f_star(n_burn+1:end);
        noise = noise(n_burn+1:end);
    case 'MASV'
        q=3;
        x=zeros(n,1);
        x(1)=0;
        noise=randn(n,1);
        a=0.5/q;
        var_ev=0.2;
        for j=2:q
            var_ev=var_ev+a*noise(j-1)^2;
            x(j)=sqrt(var_ev)*noise(j);
        end
        for j=q+1:n
            var_ev=var_ev+a*(noise(j-1)^2-noise(j-q)^2);
            x(j)=sqrt(var_ev)*noise(j);
        end

    case 'Garch'
        
        n_burn = 500; 
        N_tot = n + n_burn;
        
        omega = 0.05;
        alpha = 0.15;
        beta  = 0.80; 
        
        x_tot = zeros(N_tot, 1);
        var_ev_tot = zeros(N_tot, 1);
        noise_tot = randn(N_tot, 1);
        
        var_ev_tot(1) = omega / (1 - alpha - beta);
        x_tot(1) = sqrt(var_ev_tot(1)) * noise_tot(1);
        
        for j = 2:N_tot
            var_ev_tot(j) = omega + alpha * x_tot(j-1)^2 + beta * var_ev_tot(j-1);
            x_tot(j) = sqrt(var_ev_tot(j)) * noise_tot(j);
        end
        
        x = x_tot(n_burn+1:end);
        noise = noise_tot(n_burn+1:end);
        
        var_ev = var_ev_tot(n_burn+1:end);

    case 'Kalman'
        
        n_burn = 500;
        N_tot = n + n_burn;
        
        phi = 0.8;
        sig_w = 0.4; % State noise sd
        sig_v = 0.4; % Observation sd
        
        s_tot = zeros(N_tot, 1);
        x_tot = zeros(N_tot, 1);
        
        s_tot(1) = (sig_w / sqrt(1 - phi^2)) * randn();
        x_tot(1) = s_tot(1) + sig_v * randn();
        
        for j = 2:N_tot
            s_tot(j) = phi * s_tot(j-1) + sig_w * randn();
            x_tot(j) = s_tot(j) + sig_v * randn();
        end
        
        P_pred = sig_w^2 / (1 - phi^2); 
        for j = 1:500
            K = P_pred / (P_pred + sig_v^2); % Kalman Gain
            P_upd = (1 - K) * P_pred;
            P_pred = phi^2 * P_upd + sig_w^2;
        end
        
        K_steady = P_pred / (P_pred + sig_v^2);
        Innovation_var = P_pred + sig_v^2;

        s_hat = zeros(N_tot, 1);
        e_tot = zeros(N_tot, 1); % Raw innovations
        
        s_hat(1) = 0; % Prior mean
        for j = 1:N_tot
            e_tot(j) = x_tot(j) - s_hat(j);
            s_upd = s_hat(j) + K_steady * e_tot(j);
            if j < N_tot
                s_hat(j+1) = phi * s_upd;
            end
        end
        
        
        noise_tot = e_tot / sqrt(Innovation_var);
        
        
        x = x_tot(n_burn+1:end);
        noise = noise_tot(n_burn+1:end);
        

        P_kalman = s_hat(n_burn+1:end);  % predicted values \hat{x}_{t|t-1}
        epsi = e_tot(n_burn+1:end);      % unscaled innovations

end
