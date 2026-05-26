function [a,b] = fg(F,G,eps,b)
% Maximizes the correlation between f and g, where f = Fa, g=Gb.

[n,~]=size(F); %[~,my]=size(G);

lambda = 0.01 * (size(F,2) / n);
A=F'*G;
FF = F'*F + lambda*n*eye(size(F,2));
GG = G'*G + lambda*n*eye(size(G,2));

notdone=true;

while notdone
    ng=b'*GG*b/n;
    a=(FF\(A*b))/ng;
    nf=a'*FF*a/n;
    b_new=(GG\(A'*a))/nf;

    notdone=norm(b_new-b)/norm(b) > eps;
    b=b_new;
end

f=F*a; nf=sqrt(f'*f/n);  a=a/nf; b=b*nf;

end