function [F, Fz]=F_and_Fz(z,do_grad,choice)
% Matrix F of functions of z and its derivative.

standardize=true;
sq_max=20; %BS bound for squared x
[n, d]=size(z);

if standardize && n>1
    sig_z=std(z, 1, 1)+1e-8;
    z=(z-mean(z, 1)) ./ sig_z;
end

switch choice

    case 'oneD'

        eps=0.01;

        F=[z z.^2 (eps^2+z.^2).^(1/2)];
        if(do_grad)
            Fz=[0*z+1 2*z z./(eps^2+z.^2).^(1/2)];
        else
            Fz=0;
        end

    case 'oneD_simplest'

        % F=[z 0.1*z.^2];
        % if(do_grad)
        %     Fz=[0*z+1 0.1*2*z];
        % else
        %     Fz=0;
        % end

        F=[z z.^2];
        if(do_grad)
            Fz=[0*z+1 2*z];
        else
            Fz=0;
        end

    case 'L_and_Q'
        nq=d*(d+1)/2; % Number of quadratic terms.
        F=zeros(n,d+nq);
        Fz=zeros(n,d+nq,d);
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end


    case 'Cubic_Splines'
        % Deterministic, strictly bounded, C^2 smooth local feature space.
        % Eliminates the "tail shortcut" of polynomials and "brittle spikes" of neural nets.
        
        D=15; % Number of spline knots (controls structural resolution)
        nq=d*(d+1)/2;
        
        F=zeros(n, d+nq+D);
        Fz=zeros(n, d+nq+D, d);
        
        % --- 1. WIDE BASELINE (L and Q) ---
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d, Fz(:,j,j)=1; end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end
        
        % --- 2. CUBIC B-SPLINE KERNEL GRID ---
        % Evenly space the knots across the active state space [-3, 3]
        mu_knots=linspace(-3, 3, D);
        
        % Knot spacing determines the bandwidth of the splines
        h=mu_knots(2)-mu_knots(1); 
        
        for i=1:D
            k=k+1;
            
            % Distance from knot, scaled by spacing
            u=(z(:, 1)-mu_knots(i))/h;
            abs_u=abs(u);
            
            % Vectorized Cubic B-Spline Kernel Evaluation
            idx1=abs_u <= 1;
            idx2=(abs_u > 1) & (abs_u <= 2);
            
            F(idx1, k)=(2/3)-abs_u(idx1).^2+0.5*abs_u(idx1).^3;
            F(idx2, k)=(1/6)*(2-abs_u(idx2)).^3;
            % For |u| > 2, F remains exactly 0 (Compact Support)
            
            if(do_grad)
                sign_u=sign(u);
                
                % Exact piecewise derivative of the Cubic B-Spline
                dF_du=zeros(n, 1);
                
                dF_du(idx1)=-2*u(idx1)+1.5*sign_u(idx1).* (u(idx1).^2);
                dF_du(idx2)=-0.5*sign_u(idx2).* (2-abs_u(idx2)).^2;
                
                % Chain rule: dF/dz=dF/du*(1/h)
                Fz(:, k, 1)=dF_du/h;
            end
        end
    


    case 'Random_Fourier'
        D=100; % Number of Fourier features, higher is better approximation
        gamma=0.5;
        
        % Random sampling for weights and offset
        rng_old=rng;
        rng(42); 
        W=sqrt(2*gamma)*randn(d, D); 
        b_offset=2*pi*rand(1, D); 
        rng(rng_old);

        % Fourier feature projection: sqrt(2/D)*cos(Z*W+b)
        proj=z*W+b_offset; % (n x D) matrix
        F=sqrt(2/D)*cos(proj);
        Fz=zeros(n, D, d);
        
        if(do_grad)
            % dF/dz_j=-sqrt(2/D)*sin(Z*W+b)*W_j
            sin_proj=-sqrt(2/D)*sin(proj); % (n x D) matrix
            
            for j=1:d
                for k=1:D
                    Fz(:, k, j)=sin_proj(:, k)*W(j, k);
                end
            end
        end
    



   
    case 'Deterministic_Neural'
        % nonlinear feature space without polynomial explosion
        D=5; % number of neurons
        nq=d*(d+1)/2; 
        
        F=zeros(n, d+nq+D);
        Fz=zeros(n, d+nq+D, d);
        
        % Linear and quadratic baseline
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end
        
        % Neural expansion
        % Generate fixed, orthogonal weights using  trigonometric sequence
        W=zeros(d, D);
        for j=1:d
            W(j, :)=cos((1:D)*j*pi/D)*sqrt(2/d);
        end
        
        b_bias=sin((1:D)*pi/D); 
        
        proj=z*W+b_bias; 
        
        % ReLu smooth approximation: P/(1+exp(-P))
        sig_proj=1./(1+exp(-proj));
        F(:, d+nq+1 : end)=proj.*sig_proj;
        
        if(do_grad)
            swish_deriv=sig_proj+proj.*sig_proj.*(1-sig_proj);
            for j=1:d
                for h=1:D
                    Fz(:, d+nq+h, j)=swish_deriv(:, h)*W(j, h);
                end
            end
        end


    case 'Hermite_Expanded'
        % Orthogonal polynomials, unbounded 
        % 5th order, should bound or multiply times kernel
        nq=d*(d+1)/2;
        num_high_order=3*d; % H3, H4, H5 for each dimension
        
        F=zeros(n, d+nq+num_high_order);
        Fz=zeros(n, d+nq+num_high_order, d);
        
        % Linear and quadratic baseline
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end
        
        % Hermite 3rd to 5th degrees        
        for j=1:d
            zj=z(:,j);
            
            k=k+1;
            F(:, k)=zj.^3-3*zj;
            if(do_grad), Fz(:, k, j)=3*zj.^2-3; end
            
            k=k+1;
            F(:, k)=zj.^4-6*zj.^2+3;
            if(do_grad), Fz(:, k, j)=4*zj.^3-12*zj; end
            
            k=k+1;
            F(:, k)=zj.^5-10*zj.^3+15*zj;
            if(do_grad), Fz(:, k, j)=5*zj.^4-30*zj.^2+15; end
        end
    



    case 'Neural_Expanded'
        % Similar Deterministic_Neural, but with different parameters
        D=5; 
        nq=d*(d+1)/2; 
        
        F=zeros(n, d+nq+D);
        Fz=zeros(n, d+nq+D, d);
        
        % Linear and quadratic baseline
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end
        
        % Box-Muller pseudo-random weights generates exact N(0,1) weights
        idxW=1:(d*D);
        U1_W=max(mod(abs(sin(idxW*12.9898)*43758.5453), 1), 1e-8); 
        U2_W=mod(abs(cos(idxW*78.233)*43758.5453), 1);
        Z_W=sqrt(-2*log(U1_W)).*cos(2*pi*U2_W);
        W=reshape(Z_W, [d, D])*sqrt(2/d); 
        
        idxB=1:D;
        U1_b=max(mod(abs(sin(idxB*93.989)*43758.5453), 1), 1e-8);
        U2_b=mod(abs(cos(idxB*49.233)*43758.5453), 1);
        Z_b=sqrt(-2*log(U1_b)).*cos(2*pi*U2_b);
        b_bias=Z_b*0.1;
        

        proj=z*W+b_bias; 
        
        sig_proj=1 ./ (1+exp(-proj));
        F(:, d+nq+1 : end)=proj.* sig_proj;
        
        if(do_grad)
            swish_deriv=sig_proj+proj.* sig_proj.* (1-sig_proj);
            for j=1:d
                for h=1:D
                    Fz(:, d+nq+h, j)=swish_deriv(:, h)*W(j, h);
                end
            end
        end
    
    case 'Neural_Expanded1'
        % Same as above but weights generated randomly with fixed rng
        D=max(d*3, 5);
        nq=d*(d+1)/2; 
        
        F=zeros(n, d+nq+D);
        Fz=zeros(n, d+nq+D, d);
        
        % Linear and quadratic baseline
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=z(:,j).*z(:,l);
                if(do_grad)
                    Fz(:,k,j)=Fz(:,k,j)+z(:,l);
                    Fz(:,k,l)=Fz(:,k,l)+z(:,j);
                end
            end
        end
        
        % Random weight generation to prevent collinearity
        old_rng=rng;
        rng(38); 
        W_raw=randn(max(d, D), max(d, D));
        [Q, ~]=qr(W_raw);
        W=Q(1:d, 1:D)*sqrt(2); 
        
        b_bias=linspace(-1, 1, D);
        
        proj=z*W+b_bias; 
        rng(old_rng);

        sig_proj=1 ./ (1+exp(-proj));
        F(:, d+nq+1 : end)=proj.*sig_proj;
        
        if(do_grad)
            swish_deriv=sig_proj+proj.* sig_proj.*(1-sig_proj);
            for j=1:d
                for h=1:D
                    Fz(:, d+nq+h, j)=swish_deriv(:, h)*W(j, h);
                end
            end
        end


    case 'Gauss_Hermite_RBF'
        %Gaussian RBF with centers at roots of Hermite polynomials
        centers_per_dim=7; 
        gamma_base=0.5; 
        alpha=0.35;   
        
        % Golub-Welsch algorithm for hermite roots
        off_diag=sqrt(1:centers_per_dim-1);
        CM=diag(off_diag, 1)+diag(off_diag, -1);
        roots=sort(eig(CM)); 
        

        if d == 1
            centers=roots;
            num_centers=centers_per_dim;
        else
            grids=cell(1, d);
            [grids{:}]=ndgrid(roots);
            centers=zeros(numel(grids{1}), d);
            for j=1:d
                centers(:, j)=grids{j}(:);
            end
            num_centers=size(centers, 1);
        end
        
        F=zeros(n, num_centers);
        Fz=zeros(n, num_centers, d);
        
        for k=1:num_centers
            mu=centers(k, :); 
            dist_sq=sum((z-mu).^2, 2); 
            
            % variable bandwidth, gamma decays as the anchor moves from origin
            mu_sq=sum(mu.^2);
            gamma_k=gamma_base/(1+alpha*mu_sq);
            
            feat=exp(-gamma_k*dist_sq);
            F(:, k)=feat;
            
            if(do_grad)
                for j=1:d
                    Fz(:, k, j)=-2*gamma_k*(z(:,j)-mu(j)).*feat;
                end
            end
        end

    case 'RBF_Kernels'
        %Gaussian RBF kernels with centers at quantiles    

        centers_per_dim=7; 
        gamma_base=0.5; 
        alpha=0.35; 
        
        p=(1:centers_per_dim)/(centers_per_dim+1);
        
        %Gaussian quantiles (inverse cdf)
        q_1D=norminv(p, 0, 1)';
        
        if d == 1
            centers=q_1D;
            num_centers=centers_per_dim;
        else
            grids=cell(1, d);
            [grids{:}]=ndgrid(q_1D);
            centers=zeros(numel(grids{1}), d);
            for j=1:d
                centers(:, j)=grids{j}(:);
            end
            num_centers=size(centers, 1);
        end
        
        F=zeros(n, num_centers);
        Fz=zeros(n, num_centers, d);
        
        for k=1:num_centers
            mu=centers(k, :); 
            dist_sq=sum((z-mu).^2, 2); 
            
            mu_sq=sum(mu.^2);
            gamma_k=gamma_base/(1+alpha*mu_sq);
            
            feat=exp(-gamma_k*dist_sq);
            F(:, k)=feat;
            
            if(do_grad)
                for j=1:d
                    Fz(:, k, j)=-2*gamma_k*(z(:,j)-mu(j)).* feat;
                end
            end
        end

    
    case 'L_and_Q_bounded'
        eps=0.01;
        nq=d*(d+1)/2; % Number of quadratic terms.
        F=zeros(n,d+nq+1);
        Fz=zeros(n,d+nq+1,d);
        F(:,1:d)=z;
        if(do_grad)
            for j=1:d
                Fz(:,j,j)=1;
            end
        end
        k=d;
        for j=1:d
            for l=1:j
                k=k+1;
                F(:,k)=BS(z(:,j).*z(:,l),sq_max);
                if(do_grad)
                    denom=(sq_max^2+(z(:,j).* z(:,l)).^2).^(3/2);
                    Fz(:,k,j)=Fz(:,k,j)+(sq_max^3*z(:,l))./denom;
                    Fz(:,k,l)=Fz(:,k,l)+(sq_max^3*z(:,j))./denom;
                end
            end
        end
        F(:,k+1)=(eps^2+z(:,1).^2).^(1/2);
        Fz(:,k+1,1)=z(:,1)./(eps^2+z(:,1).^2).^(1/2);

end

end