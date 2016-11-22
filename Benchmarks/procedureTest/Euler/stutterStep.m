function [ cvnew,pR ] = stutterStep( cvLL,cvL,cv,cvR,cvRR,dt_dx,press )
%stutterStep

pL = (cvL - cvLL)./(cv - cvL);
pC = (cv - cvL)./(cvR - cv);
pR = (cvR - cv)./(cvRR - cvR);
% [(press(cv)-press(cvL))/(press(cvL)-press(cvLL)), ...
%     (press(cvR)-press(cv))/(press(cv)-press(cvL)), ...
%     (press(cvR)-press(cv))/(press(cvRR)-press(cvR))];

% pRw = [(cv-cvL)./(cvL-cvLL), ...
%     (cvR-cv)./(cv-cvL), ...
%     (cvR-cv)./(cvRR-cvR)];

tempLeft = limitor2(cvL,cv-cvL,pL);
tempRight = limitor2(cv,cv-cvR,pC);

flux = eulerFlux(tempLeft,tempRight,press);

tempLeft = limitor2(cv,cvR-cv,pC);
tempRight = limitor2(cvR,cvR-cvRR,pR);

flux = flux - eulerFlux(tempLeft,tempRight,press);
cvnew = cv + 0.25*dt_dx*flux';

end

