clear
clc
close all

x_L = 5;
dt = .01;
dv= 1024;
TH_Diff = 8.418e-5;

ds = x_L/(dv-1);
four = dt*TH_Diff/ds^2;
x_ar = 0:ds:x_L;

temper = 500*exp(-ds*(0:dv-1)/x_L);

% v = VideoWriter('HeatTrue.avi');
% v.FrameRate = 50;
% open(v);
% h1 = figure;
% set(gcf,'Visible','off');

plot(x_ar,temper,'-xr');

% g = getframe;
% writeVideo(v,g);

hold on
for k = dt:dt:1e3
   
    t1 = temper;
    title(sprintf('t = %.3f',k))
    
    temper(1) = 2*four*(t1(2)-t1(1)) + t1(1);
    temper(2:end-1) = four*(t1(1:end-2)+t1(3:end))+(1-2*four)*t1(2:end-1);
    temper(end) = 2*four*(t1(end-1)-t1(end)) + t1(end);
    
    if mod(k,200) == 0
        title(sprintf('t = %.3f',k))
        
        plot(x_ar,temper)
        drawnow;
    end
end

plot(x_ar,temper);
grid on;