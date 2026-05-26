function [Mc, M, z_sliced] = M_c(x_full, c, z_0, g, a, i0, caseH, caseF)
% Finds the derivative of M with respect to z.

[n_full,~]=size(x_full);
[mh,dz]=size(c);

% z and z_c:

z=zeros(n_full,dz);              %z
z_c=zeros(n_full,dz,mh,dz);      % dz/dc
z(1,:)=z_0;
z_ant=z(1,:);
z_c_ant=zeros(dz,mh,dz);
x_ant=x_full(1,:);

for l=2:n_full
    [H, HzC] = H_and_Hz(x_ant, z_ant, c, caseH); %H and Hz*c

    z(l,:)=tensorprod(H,c,2,1);
    z_c_new=tensorprod(HzC,z_c_ant,1,1);

    for j=1:dz
        z_c_new(j,:,j)= z_c_new(j,:,j)+H;
    end

    z_c(l,:,:,:)=z_c_new;

    z_ant=z(l,:);
    z_c_ant=z_c_new; 
    x_ant=x_full(l,:);
end

% slice now the stationary data
z_sliced = z(i0+1:end, :);
z_c_sliced = z_c(i0+1:end, :, :, :);

[F, Fz] = F_and_Fz(z_sliced,true,caseF);
F_mean = mean(F,1);
F=F-F_mean;
f=F*a; fz=tensorprod(Fz,a,2,1);
[Mz_std, M] = M_z(g,f,fz);

%Backward pass for standardization of Z

sig_z = std(z_sliced, 1, 1) + 1e-8;
z_std = (z_sliced - mean(z_sliced, 1)) ./ sig_z;

mean_Mz_std = mean(Mz_std, 1);
mean_Mz_Z = mean(Mz_std .* z_std, 1);

Mz_raw = (1 ./ sig_z) .* (Mz_std - mean_Mz_std - z_std .* mean_Mz_Z);

Mc=tensorprod(Mz_raw,z_c_sliced,[1 2],[1 2]);

end


