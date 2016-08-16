clear
clc
close all

tpb = 16;
clr = {'r','b'};
hold on
index = 1:tpb;

for k = 1:tpb/2
    for n = 1:2
        for a = k:(tpb+1)-k
            g = a+(n-1)*tpb;
            h = plot(g,k-1,strcat('o',clr{n}),'MarkerFaceColor',clr{n},'LineWidth',3,'MarkerSize',10);
            if a<(k+2) || a>(tpb-(k+1))
                set(h,'MarkerEdgeColor','k')
            end
        end
    end
end

xlim([-6,2*tpb+1])
ylim([-1,1.25*tpb])
plot([-6,2*tpb],[-.5,-.5],'k')
plot([-6,2*tpb],[0.5,0.5],'k')
plot([-6,2*tpb],[1.5,1.5],'k')
plot([-6,2*tpb],[2.5,2.5],'k')
plot([0,0],[-1,2*tpb],'k','Linewidth',5)
text(-5,0,'FullStep')
text(-5,1,'Flux')
text(-5,2,'Predictor')
text(-5,3,'PredictorFlux')




