function [x] = T_inv(y, z, a, b, k, g_mean, F_mean, lambda,caseF,caseG)
% Inverse X(y,z) of T(x, z). It can be given many y's at a time.
% For the time being, for clarity, just one z.

[n,d]=size(y);
alpha=(n-1)/n;

x=y;

[F, ~] = F_and_Fz(z,false,caseF);
F=F-F_mean;

[G, Gy] = G_and_Gy(y,true,caseG);

for l=1:k
    f=F*a{l};
    g=G*b{l};
    gy=Gy*b{l};
    x = x + lambda(l)*gy.*(f - alpha*(g-g_mean{l}));
    %x = x + lambda(k)*gy.*(f - alpha*(g-g_mean{l}));
end


end