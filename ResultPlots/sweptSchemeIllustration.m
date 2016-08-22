clear
clc
close all

%https://www.mathworks.com/matlabcentral/newsreader/view_thread/96493

tpb = 16;
clr = {'r','b'};
index = 1:tpb;
pos_top = [.05, .55, .9, .4];
pos_bottom = [.05, .05, .9, .4];

ax_top = subplot('position',pos_top);
ax_bottom = subplot('position',pos_bottom);

for k = 1:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            figure(1)
            subplot(ax_top)
            hold on
            g = a+(n-1)*tpb-1;
            h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
            
            %figure 1
            subplot(ax_bottom)
            hold on       
            if a<(k+2) || a>(tpb-(k+1))
                h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2);
                set(h,'MarkerEdgeColor','k')
                
            end
        end
    end
end

subplot(ax_top)
xlim([-1,2*tpb-.5])
ylim([-1,1.25*tpb])
%plot([-.5,-.5],[-1,2*tpb],'k','Linewidth',5)
subplot(ax_bottom)
xlim([-1,2*tpb-.5])
ylim([-1,1.25*tpb])
%plot([-.5,-.5],[-1,2*tpb],'k','Linewidth',5)


hold on
x = [.19,.11;.39,.47; .67,.59; .82, .9];
y = [.1,.15];


base = tpb + 2;
ht = base/2;
ht2 = tpb/2;

circ = [6:9,22:25];

for k = ht2:-1:1
    for n = 1:2
        for a = k:(base-k-1)
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,ht-k,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2);
            if k == 1 && sum(g == circ)> 0
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end
annotation('textarrow',x(1,:),y,'String','L_{0} -> R_{0}','LineWidth',1.5)
annotation('textarrow',x(2,:),y,'String','R_{0} -> L_{1}','LineWidth',1.5)
annotation('textarrow',x(3,:),y,'String','L_{1} -> R_{1}','LineWidth',1.5)
annotation('textarrow',x(4,:),y,'String','R_{1} -> L_{0}','LineWidth',1.5)

for k = 2:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            g = mod((a+(n-1)* tpb-1)-ht2,2*tpb);
            h = plot(g,ht2+(k-1),strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',2);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end



