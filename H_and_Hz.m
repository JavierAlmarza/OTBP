function [H, HzC] = H_and_Hz(x, z, c, choice)
% Matrix H of functions of z and x and its z derivative times C

dz=length(z);

switch choice

    case 'oneD_Linear'
        H=[z x];
        Hz=[0*z+1 0*x];

    case 'oneD_more'
        H=[z x x.^2];
        Hz=[0*z+1 0*x 0*x];

    case 'Linear_in_z'
        H=[1 z x BS(x.^2,20)]; 
        Hz=zeros(dz, dz+3);    
        for k=1:dz
            Hz(k,k+1)=1;       
        end
        
    case 'Linear_in_z_unbounded'
        H = [1, z, x, x.^2];   
        Hz = zeros(dz, dz+3);  
        for k = 1:dz
            Hz(k, k+1) = 1;   
        end
        
    case 'Mixed'
        H = [1, z, x, x.^2, tanh(z), tanh(x), x .* tanh(z)];
        Hz = zeros(dz, 3*dz + 4);
        
        for k = 1:dz
            % d(z_k) / d(z_k)
            Hz(k, 1 + k) = 1;
            
            % d(tanh(z_k)) / d(z_k)
            Hz(k, dz + 3 + k) = 1 - tanh(z(k))^2;
            
            % d(x*tanh(z_k)) / d(z_k)
            Hz(k, 2*dz + 4 + k) = x * (1 - tanh(z(k))^2);
        end

end

HzC=tensorprod(Hz,c,2,1);

end