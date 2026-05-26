function [Mz, M] = M_z(g,f,fz)
% Derivative of M with respect to z.

[n, dz, ~]=size(fz);
Mz=zeros(n, dz);

gn2=sum(g.^2)/n;
fn2=sum(f.^2)/n;

M=g'*f/n-(fn2*gn2)/2;


U = g/n - gn2*f/n;
U_centered = U - mean(U, 1);

% 3. Apply the pure local chain rule
for j=1:dz
    Mz(:,j) = U_centered .* fz(:,j);
end

end