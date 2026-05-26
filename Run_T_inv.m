% Runs T_inv

g_mean=cell(k,1);

for l=1:k
    g_mean{l}=mean(G*b{l},1);
end

z_s=2.0; % Value of z for which to invert T.

x_s = T_inv(y, z_s, a, b, k, g_mean, F_mean, lambda,caseF,caseG);

figure(7)
hist(x_s,20)