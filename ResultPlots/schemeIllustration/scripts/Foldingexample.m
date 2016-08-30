clear 
clc
close all

pos_top = [.075, .535, .9, .4];
pos_bottom = [.075, .075, .9, .4];

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
    
    plot(mod(leftidx(k),base),floor(leftidx(k)/base), 'ok', 'MarkerSize', 20)
    plot(mod(rightidx(k),base),floor(rightidx(k)/base), 'sk', 'MarkerSize', 20)
    text(mod(rightidx(k),base),floor(rightidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 12);
    text(mod(leftidx(k),base),floor(leftidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 12);
    
end

ylim([-.5,1.5])
xlim([-.5,base-.5])
title('\rm \it (a)')

%Now take them out of the arrays.
base= base+2;
height = base/2;

leftidx = height - floor(tid/2) + (mod(floor(tid/2),2) * base) + (mod(tid,2)) - 2;
rightidx = height + floor(tid/2) + (mod(floor(tid/2),2) * base) + (mod(tid,2));

subplot(ax_bottom)
hold on
for k = 1:length(tid)
    
    plot(mod(leftidx(k),base),floor(leftidx(k)/base), 'ok', 'MarkerSize', 20)
    plot(mod(rightidx(k),base),floor(rightidx(k)/base), 'sk', 'MarkerSize', 20)
    text(mod(rightidx(k),base),floor(rightidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 12 );
    text(mod(leftidx(k),base),floor(leftidx(k)/base),num2str(tid(k)), 'HorizontalAlignment','center', 'VerticalAlignment','middle', 'fontsize', 12 );
    
end
ylim([-.5,1.5])
xlim([-.5,base-.5])
title('\rm \it (b)')
ylabel('Shared array row index')
xlabel('Shared array column index')
