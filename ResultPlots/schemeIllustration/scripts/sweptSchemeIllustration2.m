clear
clc
close all

%Full Triangle with tier type.  Problem with full triangle, folding.
%Solution, cut out flux parts.  Proceed as before.
%Triangle to flat and back.  Maybe use numbers.

tpb = 16;
clr = {'r','b'};
index = 1:tpb;
pos_top = [.075, .55, .9, .4];
pos_bottom = [.075, .075, .9, .4];

ax_top = subplot('position',pos_top);
ax_bottom = subplot('position',pos_bottom);

for k = 1:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            subplot(ax_top)
            hold on
            g = a+(n-1)*tpb-1;
            h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        
            subplot(ax_bottom)
            hold on
            g = a+(n-1)*tpb-1;
            h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end

subplot(ax_top)
text(-5,0,'FullStep')
text(-5,1,'Flux')
text(-5,2,'Predictor')
text(-5,3,'PredictorFlux')
xlim([-6,2*tpb+1])
ylim([-1,.5*tpb])
plot([-.5,-.5],[-1,2*tpb],'k','Linewidth',3)
title(sprintf('Second order swept scheme with flux\n%-s','\rm \it (a)'))
for k = 0:3
    plot([-6,2*tpb],[k-.5,k-.5],'k')
end
subplot(ax_bottom)
xlim([-6,2*tpb+1])
ylim([-1,.5*tpb])
plot([-.5,-.5],[-1,2*tpb],'k','Linewidth',3)
title('\rm \it (b)')
xlabel('Spatial point')
ylabel('Sub-timestep')
x1 = [2,3,12,13,6:9];
y = [0,0,0,0,4,4,4,4];
hold on
plot(x1,y,'Xk','Markersize',15,'Linewidth',5)
x2 = x1 + 2*(15-x1) +1; 
plot(x2,y,'Xk','Markersize',15,'Linewidth',5)
% % some data with an awkward axis
%      plot(30:40,rand(1,11));
% % create a text annotation
%      ah=annotation('textbox');
% % ...and position it on the current axis
%      set(ah,'parent',gca);
%      set(ah,'position',[31 .1 3 .2]);

figure(2)
ax_top = subplot('position',pos_top);
ax_bottom = subplot('position',pos_bottom);


for k = 1:tpb/4
    for n = 1:2
        for a = 2*(k-1):((tpb-1)-2*(k-1))
            subplot(ax_top)
            hold on
            g = a+(n-1)*(tpb);
            h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                set(h,'MarkerEdgeColor','k')
            end
            subplot(ax_bottom)
            hold on
            g = a+(n-1)*(tpb);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end

base = tpb + 2;
ht = base/2;
ht2 = tpb/2;
circ = [4:11,20:27];
hold on
c = 1;
y = 1:4;
for k = ht2:-2:1
    for n = 1:2
        for a = k-1:(base-k)
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,y(c),strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
            if k == 2 && sum(g == circ)> 0
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
    c = c+1;
end

subplot(ax_top)
xlim([-1,2*tpb-.5])
ylim([-1,.5*tpb])
title('\rm \it (a)')

subplot(ax_bottom)
xlim([-1,2*tpb-.5])
ylim([-1,.5*tpb])

title('\rm \it (b)')
xlabel('Spatial point')
ylabel('Sub-timestep')
y = 5:8;
c = 1;
for k = 2:tpb/4
    for n = 1:2
        for a = 2*(k-1):((tpb-1)-2*(k-1))
            g = mod((a+(n-1)* tpb-1)-ht2+1,2*tpb);
            h = plot(g,y(c),strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2,'MarkerSize',10);
            if a<(2*(k-1)+4) || a>((tpb-1)-2*(k-1)-4)
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
    c = c+1;
end
