% OTBP with z representing the past of x.

% For all time-dependent arrays, the first index denotes time.

% size(c) = (mh, dz)

% Meaning of the various tensors:

% x(l,j): j'the component of the observation x at time l.
% y(l,j): j'the component of T(x(l,:), z(l,:)), where z = U_k (z{k}).

% z{k}(l, j): j'th component of z{k} at time l.

% c{k}(i,j): z{k}(l+1},j) = sum_i H(z{k}(l,:), x(l,:))[i] c{k}(i, j)

% H(l,i) = H(z{k}(l,:), x(l,:))[i]

% Hz(l,i,j): dH(l,i)/dz{k}(l,j);

% HzC(l,h,j): sum_i Hz(l,i,h) c(i,j)

% f{k}(l), g{k}(l): functions f(z{k}) and g{k}(y) at time l.
% a{k}, b{k}: f{k} = F a{k}, g{k} = G(y) b{k} 

% (b{k} and g{k} are not used, they are only kept as a record of what 
% happened in step 1 of stage k). 

% F(l,j): j'th component of F(z{k}(l))
% G(l,j): j'the component of G(y(l))

% Fz(l,j,h) = dF(l,l)/dz{k}(l,h)
% Gy(l,j,h) = dG(l,j)/dy(l,h)

% Mc(i,j): dM/dc(i,j).

% Mz(l,j); dM/dz{k}(l,j)

clear all

%Example='Markov_1';
% Example='Markov_2';
% Example='Markov_3';
% Example='MASV';
Example = 'Garch';
%Example = 'Kalman';

fprintf('Data is %s',Example)

dz=1; % Dimension of z sought.

Data_OTBP_with_past; %Reads or generates x.

% caseF='oneD';
%caseF='L_and_Q';
caseF='L_and_Q_bounded';
caseG='oneD_L_and_Q';
% caseH='oneD_Linear';
% caseH='oneD_more';
caseH='Linear_in_z';
%caseH='Linear_in_z_unbounded';

optimizer_type = 'Adam';
%optimizer_type = 'Gradient descent'

switch caseG
    case 'oneD_L'
        my=1;
    case 'oneD_L_and_Q'
        my=2;
end

[n,~]=size(x);

eta_0=0.01;
eta_min=0.001;
nsmax=4000;

switch caseH
    case 'oneD_Linear'
        cc_0=[0.5; 0.5]; 
    case 'oneD_more'
        cc_0=[0.5; 0.5; 0.5];
        
    case {'Linear_in_z', 'Linear_in_z_unbounded'} 
        cc_0 = 0.1 * rand(dz+3, dz);   
        cc_0(1, :) = 0.01 * rand();                
        cc_0(dz+2, :) = 0.5 * rand(1, dz); 
        
    case 'Linear_in_z_bounded'
        cc_0=0.1*rand(dz+2,dz);
        cc_0(dz+1,:)=0.5*rand(1,dz);
end

i0=100; % Warm-up period for z.
z_0=zeros(1,dz); % Fixed z at time 0.

sigma_0 = max(0.001,0.1/sqrt(n)); % Threshold of acceptable correlations.
eps=0.0001; % Termination criterion for step 1.
Lymin=eps/sqrt(n);

Kmax=5;
Cpr=zeros(1,nsmax*Kmax); Ppr=Cpr; etapr=Cpr; Lampr=zeros(Kmax,nsmax*Kmax);
sigma=zeros(1,Kmax); lambda=sigma;

f=cell(Kmax,1); g=f; a=f; b=f; c=f; P=f; Py=f; % sigma=f;

y=x;

% First step 1:

% z=zeros(n,dz);
%z=0.1*rand(n,dz);

% z(2:n,1)=x(1:n-1,:);
%z(1,:)=z_0;

% z = Find_z(x, cc_0, z_0, caseH);
% 
% [F, ~] = F_and_Fz(z,false,caseF); % Matrix F.
% F_mean = mean(F,1);
% F=F-F_mean;
% 
% [G, ~] = G_and_Gy(y,false,caseG); % G.
% G_mean = mean(G,1);
% G=G-G_mean;

k=0;

%Loop over stages.

not_done_yet=true;
jc=0;

while not_done_yet

    z = Find_z(x, cc_0, z_0, caseH);
    z = z + 0.1*randn(n,dz);

    cc=cc_0;

    [F, ~] = F_and_Fz(z,false,caseF); % Matrix F.
    F_mean = mean(F,1);
    F=F-F_mean;

    % eig(F'*F)
    % stop

    [G, ~] = G_and_Gy(y,false,caseG); % G.
    G_mean = mean(G,1);
    G=G-G_mean;

    not_done_yet_1=true;

    bb=rand(my,1)-1/2;

    [aa,bb] = fg(F,G,eps,i0,bb);
    gg=G*bb;

    [Mc, M_old, z_old] = M_c(x, cc, z_0, gg, aa, i0, caseH, caseF);
    Mc2=sum(sum(Mc.^2));

    if strcmp(optimizer_type, 'Adam')
        m_adam = zeros(size(cc));
        v_adam = zeros(size(cc));
        t_adam = 0;
        beta1 = 0.9; beta2 = 0.999; eps_adam = 1e-8;
    end
    eta=eta_0;

    while not_done_yet_1

        if strcmp(optimizer_type, 'Adam')
            t_adam = t_adam + 1;
            m_adam = beta1 * m_adam + (1 - beta1) * Mc;
            v_adam = beta2 * v_adam + (1 - beta2) * (Mc.^2);
            m_hat = m_adam / (1 - beta1^t_adam);
            v_hat = v_adam / (1 - beta2^t_adam);
            
            step_dir = m_hat ./ (sqrt(v_hat) + eps_adam);
        else
            step_dir = Mc;
        end
        
        cc = cc + eta * step_dir;

        [F, ~] = F_and_Fz(z_old,false,caseF); 
        F_mean = mean(F,1);
        F=F-F_mean;

        [aa,bb] = fg(F,G,eps,i0,bb);
        gg=G*bb;

        [Mc, M_new, z_new] = M_c(x, cc, z_0, gg, aa, i0, caseH, caseF);
        Mc2=sum(sum(Mc.^2));

        if(M_new-M_old > 0.5*eta*Mc2)
            eta=1.01*eta;
        else
            eta=0.81*eta;
        end

        delta_z=norm(z_new-z_old)/norm(z_old);

        M_old=M_new;
        if(delta_z < eps)
            not_done_yet_1=false;
            z=z_new;
        else
            z_old=z_new;
        end
    end

    [F, ~] = F_and_Fz(z,false,caseF);
    F_mean = mean(F,1);
    F=F-F_mean;

    ff=F*aa;
    if(abs(corr(ff,gg)) < 2.*sigma_0)
        not_done_yet=false;
    else
        nf=sqrt(ff'*ff/n);
        k=k+1;
        a{k}=aa/nf;
        f{k}=ff/nf;
        b{k}=bb;
        c{k}=cc;
        
        
        % GARCH theoretical correlation check    
        if strcmp(Example, 'Garch') && k == 1
            actual_corr = corr(ff, gg);
            
            c_true = zeros(size(cc_0)); 
            
            c_true(1, 1)    = 0.05; % omega 
            c_true(2, 1)    = 0.80; % beta  
            c_true(dz+2, 1) = 0.00; % 0 coeff for x
            c_true(dz+3, 1) = 0.15; % alpha 
            

            z_true = Find_z(x, c_true, z_0, caseH);
            [F_true, ~] = F_and_Fz(z_true, false, caseF);
            F_true = F_true - mean(F_true, 1);
            
            % Solve for the optimal a and b weights using true F and current G
            b_init = rand(my, 1) - 0.5;
            [a_true, b_true] = fg(F_true, G, eps, i0, b_init);
            
            theo_corr = corr(F_true * a_true, G * b_true);
            
            fprintf('\n- stage 1 step 1 correlation diagnostics -\n');
            fprintf('Obtained correlation   : %.5f\n', actual_corr);
            fprintf('Theoretical GARCH correlation : %.5f\n', theo_corr);
            fprintf('---------------------------------------------\n\n');
        end
    end

    if(not_done_yet)

        % Step 2:

        eta=eta_0;

        y_not_converged=true;

        js=0;

        while y_not_converged && js < nsmax

            js=js+1;
            jc=jc+1;

            etapr(jc)=eta;

            [G, Gy] = G_and_Gy(y,true,caseG); % G and Gy.
            G_mean = mean(G,1);
            G=G-G_mean;

            Ly = (y-x)/n; % Gradient of the quadratic cost.
            LL=0.;
            L_old=sum(sum((y-x).^2))/n;

            % Gradient of the penalty terms.
            for l=1:k
                [P{l}, Py{l}, b{l}, sigma(l)] = P_and_Py_l(G,Gy,f{l},true);
            end

            % Calculation of the lambdas and Ly:
            [sigmas,Isigmas]=sort(sigma(1:k),'ascend');
            sigmas=min(sigmas/sigma_0-0.8,30.);
            for l=1:k
                c_uv=sum(sum(Ly.*Py{Isigmas(l)}))/sum(sum(Py{Isigmas(l)}.^2));
                c_uv=min(c_uv,0);
                lambda(l)=max(sigmas(l)-c_uv,0);
                Ly=Ly+lambda(l)*Py{Isigmas(l)};
                LL=LL+lambda(l)*P{Isigmas(l)};
            end
            lambda(1:k)=lambda(Isigmas); % Undo the sorting

            Lampr(:,jc)=lambda;
            Cpr(jc)=L_old;
            L_old=L_old+LL;
            Ppr(jc)=LL;

            Lys=sum(sum(Ly.^2));

            y = y - eta*Ly;
            L_new=sum(sum((y-x).^2))/n;

            [G, ~] = G_and_Gy(y,false,caseG);
            G_mean = mean(G,1);
            G=G-G_mean;

            Smax=-1;
            LL=0;
            for l=1:k
                [P{l}, ~, ~, sigma(l)] = P_and_Py_l(G,0,f{l},false);
                LL=LL+lambda(l)*P{l};
                Smax=max(Smax,sigma(l));
            end
            L_new=L_new+LL;

            if(L_old-L_new > 0.5*eta*Lys-eps)
                eta=1.01*eta;
            else
                eta=0.81*eta;
            end

            y_not_converged=(Smax > sigma_0 || norm(Ly)/sqrt(n) > Lymin);

            % y_not_converged=(Smax > sigma_0);

        end

    end

    if(k == Kmax)
        not_done_yet=false;
    else
        [G, ~] = G_and_Gy(y,false,caseG); % G.
        G_mean = mean(G,1);
        G=G-G_mean;
    end
end

% figure(1)
% plot(z,x,'b.')
% hold on
% plot(z,y,'r.')
% xlabel('z')
% ylabel(['x and y'])
% hold off

figure(2)
plot(Cpr(1:jc),'b')
hold on
plot(Ppr(1:jc),'r')
ylabel('Cost and Penalty')
hold off

figure(3)
plot(etapr(1:jc),'g')
ylabel('\eta')

figure(4)
plot(Lampr(1:k,1:jc)','g')
ylabel('\lambda')

figure(5)
plot(noise,x,'b*')
xlabel('noise')
ylabel('x')

figure(6)
plot(noise,y,'r*')
xlabel('noise')
ylabel('y')

if strcmp(Example, 'Garch')
    sigma_t = sqrt(var_ev); 
    
    eff_MSE = mean((x - y).^2) / 2; 
    predicted_val = var(sigma_t, 1) / 2; 
    
    E_sigma = mean(sigma_t);
    y_star = noise .* E_sigma;
    RSE = sum((y - y_star).^2) / sum(y_star.^2);
    
    fprintf('Final MSE ||x-y||^2/(2T) : %.4f\n', eff_MSE);
    fprintf('Predicted Equilibrium Value  : %.4f\n', predicted_val);
    fprintf('Relative Sqd Error for Y : %.4f\n', RSE);
end

if strcmp(Example, 'Kalman')
    y_star = epsi; 
    
    eff_MSE = mean((x - y).^2) / 2; 
    predicted_val = mean(P_kalman.^2) / 2; 
    
    RSE = sum((y - y_star).^2) / sum(y_star.^2);
    
    fprintf('Final MSE ||x-y||^2/(2T) : %.4f\n', eff_MSE);
    fprintf('Relative Sqd Error for Y : %.4f\n', predicted_val);
    fprintf('Relative Squared Error for y : %.4f\n\n', RSE);
end
