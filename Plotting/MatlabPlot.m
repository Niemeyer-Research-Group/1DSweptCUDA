clear
clc
close all

x = dlmread('1DSweptTiming.txt','\t',1,0);

funx = scatteredInterpolant(x(:,1),x(:,2),x(:,3));

blk = unique(x(:,1))';
div = unique(x(:,2))';
Leg = {};
p1 = {'--','-o','-.','*-',':s',':'};

for k = 1:length(blk)
    Leg{k} = sprintf('BlockSize: %.f',blk(k));
    for n = 1:length(div)
        G(n) = funx(blk(k),div(n));
    end
    figure(1)
    semilogx(div,G,strcat('k',p1{k}))
    hold on
    figure(2)
    loglog(div,G,strcat('k',p1{k}))
    hold on
end
legend(Leg,'location','Northwest','FontSize',12)
ylabel('Calculation Time (s)','FontSize',12)
xlabel('Number of spatial points','FontSize',12)
figure(1)
legend(Leg,'location','Northwest')
ylabel('Calculation Time (s)')
xlabel('Number of spatial points')

