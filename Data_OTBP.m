% Data generation (z and x).

n=2000; % Number of samples

signal=4*(rand(n,1)-0.5);

z=signal;
noise=randn(n,1);
x=z + 0.1*(4*z.^2+0.5).^(1/2).*noise;