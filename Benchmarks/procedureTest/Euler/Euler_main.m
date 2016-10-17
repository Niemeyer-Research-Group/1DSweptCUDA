clear
clc
close all

bd = [1, 0, 1/0.4; 0.125, 0, 0.1/0.4];
bd = bd';

xrng = 1024;

temper = zeros(3,xrng+4);
pR = temper;

for k = 1:length(temper)
    if k<=((xrng+4)/2)+1
        temper(:,k) = bd(:,1);
    else
        temper(:,k) = bd(:,2);
    end
end

temper2 = temper;

press = @(current) 0.4*(current(3)-(0.5*current(2)^2/current(1)));

x = linspace(0,1,xrng+1);
dx = x(2);
x = dx*0.5 + x;
x = [-dx*1.5, -dx*0.5, x, 1+dx*1.5];
dt= 5e-5;
dtdx = dt/dx;

ti = {'Density','Velocity','Energy'};

t = dt;
ts = .15;
t2 = 0.05;

tic
while t<ts
    
   for k = 3:length(temper)-2
       
       [temper2(:,k),pR(:,k)] = stutterStep(temper(:,k-2),temper(:,k-1),temper(:,k), ...
           temper(:,k+1),temper(:,k+2),dtdx,press);
       
       temper(:,k) = temper(:,k) + fullStep(temper2(:,k-2),temper2(:,k-1), ...
           temper2(:,k),temper2(:,k+1),temper2(:,k+2),dtdx,press);
       
   end
   t = t+dt;
   
end
toc

p = zeros(1,xrng);
for k = 3:length(temper)-2
    p(k-2) = press(temper(:,k));
end

temper(2:3,:) = [temper(2,:)./temper(1,:);temper(3,:)./temper(1,:)]; 

for k = 1:3
    subplot(2,2,k)
    
    plot(x(3:end-2),temper(k,3:end-2));
    title(strcat(ti{k},' at t = ',num2str(t)))
    grid on
end

subplot(2,2,4)

plot(x(3:end-2),p);
title(strcat('Pressure at t = ',num2str(t)));
grid on

