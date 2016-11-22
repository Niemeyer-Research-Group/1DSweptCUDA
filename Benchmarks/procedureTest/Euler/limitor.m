function [ cvNow ] = limitor( cvCurrent, cvOther, pRatio )
%limitor


if (~isnan(pRatio) && pRatio > 0)
    fact = 0.5*max([min([2*pRatio, 1]),min([pRatio,2])]);
    cvNow = cvCurrent + fact * (cvOther - cvCurrent);
else
    cvNow = cvCurrent;
end

end

