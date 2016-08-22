clear 
clc
close all

pos_top = [.05, .55, .9, .4];
pos_bottom = [.05, .05, .9, .4];

ax_top = subplot('position',pos_top);
ax_bottom = subplot('position',pos_bottom);

%This is right, but they do overlap when you put them IN the arrays.
subplot(ax_top)
tid = (1:16)-1;
base = 16;
leftidx = floor(tid/2) + (mod(floor(tid/2),2) * base) + mod(tid,2);
rightidx = (base - 2) + (mod(floor(tid/2),2) * base) + mod(tid,2) -  floor(tid/2);
hold on
for k = 1:base
    
    plot(mod(leftidx(k),base),floor(leftidx(k)/base), 'or', 'MarkerSize', 15)
    plot(mod(rightidx(k),base),floor(rightidx(k)/base), 'ob', 'MarkerSize', 16)
    text(mod(rightidx(k),base),floor(rightidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 8 );
    text(mod(leftidx(k),base),floor(leftidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 8 );
    
end
grid on
ylim([-1,2])
xlim([-1,base+1])
title('Index for extracting edges to global arrays')

%Now take them out of the arrays.
base= base+2;
height = base/2;

leftidx = height - floor(tid/2) + (mod(floor(tid/2),2) * base) + (mod(tid,2)) - 2;
rightidx = height + floor(tid/2) + (mod(floor(tid/2),2) * base) + (mod(tid,2));

subplot(ax_bottom)
hold on
grid on
for k = 1:length(tid)
    
    plot(mod(leftidx(k),base),floor(leftidx(k)/base), 'or', 'MarkerSize', 16)
    plot(mod(rightidx(k),base),floor(rightidx(k)/base), 'ob', 'MarkerSize', 16)
    text(mod(rightidx(k),base),floor(rightidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 8 );
    text(mod(leftidx(k),base),floor(leftidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 8 );
    
end
ylim([-1,2])
xlim([-1,base+1])
title('Reinserting left and right into shared array')