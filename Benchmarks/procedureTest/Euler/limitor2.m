function [ cvNow ] = limitor2( cvCurrent, cvsub, pRatio )
%limitor2

for k = 1:3
    if (~isnan(pRatio(k)) && pRatio(k) > 0)
        fact = min([1, pRatio(k)/(1+pRatio(k))]);
        cvNow(k) = cvCurrent(k) + fact * cvsub(k);
    else
        cvNow(k) = cvCurrent(k);
    end

end

