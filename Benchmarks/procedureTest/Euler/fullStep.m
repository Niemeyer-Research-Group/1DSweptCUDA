function [ rh ] = fullStep( cvLL,cvL,cv,cvR,cvRR,dt_dx,press )
%fullStep 

pR = [(press(cv)-press(cvL))/(press(cvL)-press(cvLL)), ...
    (press(cvR)-press(cv))/(press(cv)-press(cvL)), ...
    (press(cvR)-press(cv))/(press(cvRR)-press(cvR))];

pRw = [(cv-cvL)./(cvL-cvLL), ...
    (cvR-cv)./(cv-cvL), ...
    (cvR-cv)./(cvRR-cvR)];

tempLeft = limitor(cvL,cv,pR(1),pRw(:,1));
tempRight = limitor(cv,cvL,1/pR(2), 1./pRw(:,2));

flux = eulerFlux(tempLeft,tempRight,press);

tempLeft = limitor(cv,cvR,pR(2),pRw(:,2));
tempRight = limitor(cvR,cv,pR(3),pRw(:,3));

flux = flux - eulerFlux(tempLeft,tempRight,press);

rh = 0.5*dt_dx*flux';

end

