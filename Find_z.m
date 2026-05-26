function z = Find_z(x, c, z_0, caseH)
% Finds z.

[n,~]=size(x);
[~,dz]=size(c);

z=zeros(n,dz);
z(1,:)=z_0;
z_ant=z(1,:);
x_ant=x(1,:);

for l=2:n
    [H, ~] = H_and_Hz(x_ant, z_ant, c, caseH); %H and Hz*c
    z(l,:)=tensorprod(H,c,2,1);
    z_ant=z(l,:);
    x_ant=x(l,:);
end


