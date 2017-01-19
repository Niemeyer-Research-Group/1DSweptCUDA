clear
clc
close all

% Boundary conditions
bd = [1, 0, 1/0.4; 0.125, 0, 0.1/0.4];
bd = bd';

% Spatial grid and timestep
xrng = 512;
dt= 2e-5;
x = linspace(0,1,xrng);
dx = x(2);

% Timegrid and dtdx.
dtdx = dt/dx;
fprintf('dt_dx = %.4f\n',dtdx)
ts = 0.1;
ta = 0:dt:ts;
ti = {'Density','Velocity','Energy'};

% The working array
temper = zeros(3,xrng,2);

% The functions to execute this thing.
press = @(current) 0.4*(current(3)-(0.5*current(2)^2/current(1)));

Fw = @(current) [current(2) ; ...
    current(2)^2/current(1) + press(current); ...
    current(2)/current(1)*(current(3)+press(current))];

halfSol = @(cL, cR) 0.5*((cL + cR) + dtdx*(Fw(cR)-Fw(cL)));

%Set up the array with the initial conditions.
for k = 1:length(temper)
    if k<=((xrng)/2)
        temper(:,k,1) = bd(:,1);
        temper(:,k,2) = bd(:,1);
    else
        temper(:,k,1) = bd(:,2);
        temper(:,k,2) = bd(:,2);
    end
end
disp(length(ta));
%And this is the main loop.  Outside is timestep, inside is across space.
tic
for n = 2:length(ta)
    nn = mod(n,2) + 1;
    nm = mod(nn,2) + 1;
    for k = 2:xrng-1
        temper(:,k,nm) = temper(:,k,nn) - dtdx*(Fw(halfSol(temper(:,k,nn), ...
            temper(:,k+1,nn)))- Fw(halfSol(temper(:,k-1,nn),temper(:,k,nn))));
       
    end

    if mod(n,50) == 0
        disp(n)
    end
    
end
toc
temper = temper(:,:,nm);

p = zeros(1,xrng);
for k = 1:length(temper)
    p(k) = press(temper(:,k));
end

temper(2:3,:) = [temper(2,:)./temper(1,:); (temper(3,:)./temper(1,:))];

for k = 1:3
    subplot(2,2,k)
    
    plot(x,temper(k,:));
    title(strcat(ti{k},' at t = ',num2str(ts)))
    grid on
end

subplot(2,2,4)

plot(x,p);
title(strcat('Pressure at t = ',num2str(ts)));
grid on

