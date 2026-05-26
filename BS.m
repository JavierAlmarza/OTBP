function y = BS(x,a)
% A version of x with |x| bounded by a.

y = a*x./(sqrt(a^2 + x.^2));

% dy/dx = a^3/(a^2+x^2)^(3/2)

end