function [ cvnew ] =stutterStep( cvLL,cvL,cv,cvR,cvRR,dt_dx,press )
%stutterStep Summary of this function goes here
%   Detailed explanation goes here

flux = eulerFlux(cvL,cv,press);
flux = flux - eulerFlux(cv,cvR,press);
cvnew = cv + 0.25*flux'*dt_dx;

end

