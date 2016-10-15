function [ rh ] = fullStep( cvLL,cvL,cv,cvR,cvRR,dt_dx,press )
%fullStep 

flux = eulerFlux(cvL,cv,press);
flux = flux - eulerFlux(cv,cvR,press);

rh = 0.5*dt_dx*flux';

end

