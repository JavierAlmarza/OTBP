% OTBP

clear all

Data_OTBP; %Reads or generates x and z.
% Data_OTBP_with_past;

caseF='oneD';
% caseG='oneD_L_and_Q';
caseG='oneD_L_and_Q';

switch caseG
    case 'oneD_L_and_Q'
        my=2;
    case 'oneD_L'
        my=1;
end


[n,~]=size(x);
% sx=std(x,[],'all')/sqrt(n);
eta_0=0.1;
eta_min=0.00001;
nsmax=1000;

[F, ~] = F_and_Fz(z,false,caseF); % Matrix F.
F_mean = mean(F,1);
F=F-F_mean;

sigma_0 = max(0.001,0.1/sqrt(n)); % Threshold of acceptable correlations.
eps=0.0001; % Termination criterion for step 1.
Lymin=eps/sqrt(n);

Kmax=3;
Cpr=zeros(1,nsmax*Kmax); Ppr=Cpr; etapr=Cpr; Lampr=zeros(Kmax,nsmax*Kmax);
sigma=zeros(1,Kmax); lambda=sigma;

f=cell(Kmax,1); g=f; a=f; b=f; P=f; Py=f; % sigma=f;

y=x;

[G, ~] = G_and_Gy(y,false,caseG); % G.
G_mean = mean(G,1);
G=G-G_mean;

k=1;

% First step 1:

bb=rand(my,1)-1/2;
[a{k},b{k}] = fg(F,G,eps,0,bb);
f{k}=F*a{k}; % k'th function f^k(z), which will remain fixed through the algorithm.
g{k}=G*b{k}; % k'th function g^k(y). Both y and $g^k will evolve.
% c=corr(f{k},g{k});

%Loop over stages.

not_done_yet=true;
jc=0;

while not_done_yet

    % Step 2:

    y_not_converged=true;
%    not_there_yet=true;

    js=0;
    eta=eta_0;

    while y_not_converged && js < nsmax

        js=js+1;
        jc=jc+1;

        etapr(jc)=eta;

        [G, Gy] = G_and_Gy(y,true,caseG); % G and Gy.
        G_mean = mean(G,1);
        G=G-G_mean;

        Ly = (y-x)/n; % Gradient of the quadratic cost.
        LL=0.;
        LLy=0.;
        H=0.;
        L_old=sum(sum((y-x).^2))/n;

        % Gradient of the penalty terms.
        for l=1:k
            [P{l}, Py{l}, b{l}, sigma(l)] = P_and_Py_l(G,Gy,f{l},true);
        end

        % Calculation of the lambdas and Ly:
        [sigmas,Isigmas]=sort(sigma(1:k),'ascend');
        sigmas=min(sigmas/sigma_0-0.9,20.);
        for l=1:k
            c_uv=sum(sum(Ly.*Py{Isigmas(l)}))/sum(sum(Py{Isigmas(l)}.^2));
            c_uv=min(c_uv,0);
            lambda(l)=max(sigmas(l)-c_uv,0);
            Ly=Ly+lambda(l)*Py{Isigmas(l)};
            LL=LL+lambda(l)*P{Isigmas(l)};
        end
        lambda(1:k)=lambda(Isigmas); % Undo the sorting

        Lampr(:,jc)=lambda;
        Cpr(jc)=L_old;
        L_old=L_old+LL;
        Ppr(jc)=LL;

        Lys=sum(sum(Ly.^2));

        y = y - eta*Ly;
        L_new=sum(sum((y-x).^2))/n;

        [G, ~] = G_and_Gy(y,false,caseG);
        G_mean = mean(G,1);
        G=G-G_mean;

        Smax=-1;
        LL=0;
        for l=1:k
            [P{l}, ~, ~, sigma(l)] = P_and_Py_l(G,0,f{l},false);
            LL=LL+lambda(l)*P{l};
            Smax=max(Smax,sigma(l));
        end
        L_new=L_new+LL;

        if(L_old-L_new > 0.5*eta*Lys-eps)
            eta=1.01*eta;
        else
           eta=0.81*eta;
        end

    %    not_there_yet=Smax > sigma_0;

        y_not_converged=(Smax > sigma_0 || norm(Ly)/sqrt(n) > Lymin);

    end

    % Step 1:

    bb=rand(my,1)-1/2;
    [aa,bb] = fg(F,G,eps,0,bb);
    ff=F*aa; % k'th function f^k(z).
    gg=G*bb;
    c=corr(ff,gg);
    if(c < 2.5*sigma_0)
        not_done_yet=false;
    else
        if(k < Kmax)
            k=k+1;
            a{k}=aa;
            f{k}=ff;
        else
            not_done_yet=false;
        end
    end

end

figure(1)
plot(z,x,'b.')
hold on
plot(z,y,'r.')
xlabel('z')
ylabel(['x and y'])
hold off

figure(2)
plot(Cpr(1:jc),'b')
hold on
plot(Ppr(1:jc),'r')
ylabel('Cost and Penalty')
hold off

figure(3)
plot(etapr(1:jc),'g')
ylabel('\eta')

figure(4)
plot(Lampr(:,1:jc)','g')
ylabel('\lambdas')

figure(5)
plot(noise,x,'b*')
xlabel('noise')
ylabel('x')

figure(6)
plot(noise,y,'r*')
xlabel('noise')
ylabel('y')


