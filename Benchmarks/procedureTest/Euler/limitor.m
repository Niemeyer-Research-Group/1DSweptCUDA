function [ cvNow ] = limitor( cvCurrent, cvOther, pRatio )
%limitor

if (~isnan(pRatio) && pRatio > 0)
    fact = min([pRatio*0.5, 0.5]);
    cvNow = cvCurrent + fact * (cvOther - cvCurrent);
else
    cvNow = cvCurrent;
end

end

