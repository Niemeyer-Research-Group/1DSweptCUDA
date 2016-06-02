clear
clc
close all

x = dlmread('1DSweptTiming.txt','\t',1,0);

funx = scatteredInterpolant(x(:,1),x(:,2),x(:,3));

blk = unique(x(:,1))';
div = unique(x(:,2))';
Leg = {};

for k = 1:length(blk)
    Leg{k} = strcat('BlkSz: ',num2str(blk(k)));
    for n = 1:length(div)
        G(n) = funx(blk(k),div(n));
    end
    semilogx(div,G)
    hold on
end
legend(Leg)

