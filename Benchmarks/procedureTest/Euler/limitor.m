function [ cvNow ] = limitor( cvCurrent, cvOther, pRatio, cvR )
%limitor

if (~isnan(pRatio) && pRatio > 0)
    fact = min([0.5*cvR; 0.5, 0.5, 0.5]);
    cvNow = cvCurrent + fact' * (cvOther - cvCurrent);
else
    cvNow = cvCurrent;
end

end

