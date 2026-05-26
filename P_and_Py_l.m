function [P, Py, b, sigma] = P_and_Py_l(G,Gy,f,do_grad)
%Individual penalty P_l and its y-derivative.
% P = 1/n [f' Gb - 1/2 b' G'G b] 

n=length(f);

b = ((G'*G)\G')*f;

g = G*b; gy=Gy*b; 

P = (f-g/2)'*g/n;   

if(do_grad)
    Py = (f - g).*gy/n;
else
    Py=0;
end

sigma=corr(f,g);

% Verify things:
% 
% PP=(corr(f,g))^2/2;
% 
% cor2=(f'*g)/(sqrt(n)*norm(g));
% 
% PP2=cor2^2/2;
% 
% [P PP PP2]
% 
% PPy7=corr(f,g)*(f(7)/(sqrt(n)*norm(g))-corr(f,g)*g(7)/norm(g)^2)*gy(7);
% 
% [Py(7) PPy7]
% 
% stop


end