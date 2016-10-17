function [ flux ] = eulerFlux( cvL,cvR,press )
%eulerFlux 

uL = cvL(2)/cvL(1);
uR = cvR(2)/cvR(1);
eL = cvL(3)/cvL(1);
eR = cvR(3)/cvR(1);

rRsqrt = sqrt(cvR(1));
rLsqrt = sqrt(cvL(1));

pR = press(cvR);
pL = press(cvL);

halfstate(1) = rRsqrt*rLsqrt;
halfstate(2) = (rRsqrt*uR + rLsqrt*uL)/(rRsqrt+rLsqrt);
halfstate(3) = (rRsqrt*eR + rLsqrt*eL)/(rRsqrt+rLsqrt)*halfstate(1);

halfstate2 = halfstate.*[1, halfstate(1), 1];
pH = press(halfstate2);

flux(1) = (cvL(2) + cvR(2));
flux(2) = (cvL(2)*uL + cvR(2)*uR + pR + pL); 
flux(3) = (cvL(2)*eL + cvR(2)*eR + pR*uR + pL*uL);

flux = flux + (sqrt(1.4*pH/halfstate(1))+abs(halfstate(2))) * (cvL'-cvR');


end

