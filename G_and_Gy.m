function [G, Gy] = G_and_Gy(y, do_grad, choice)
% Matrix F of functions of z and its derivative.

switch choice

    case 'oneD_L_and_Q'

        n=length(y);

        G=[y y.^2];

        if(do_grad)
            %Gy=(1-1/n)*[0*y+1 2*y];
            Gy=[0*y+1 2*y];
        else
            Gy=0;
        end

    case 'oneD_L'

        n=length(y);

        G=[y];

        if(do_grad)
            %Gy=(1-1/n)*[0*y+1];
            Gy=[0*y+1];
        else
            Gy=0;
        end

end

end