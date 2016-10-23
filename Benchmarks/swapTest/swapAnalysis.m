clear
clc
close all

fy = dir('*.txt');
format long g

files = {fy.name};

names = {files{1}(11:end-4)};
data = dlmread(files{1});

for k = 2:length(files)
    names{k} = files{k}(11:end-4);
    data(:,:,k) = dlmread(files{k});
end

[x,y,z] = size(data);

sz = unique(data(:,2,1));
typ = unique(data(:,1,1));
tpb = unique(data(:,3,1));
rsltrow = zeros(length(sz),y);

Leg = {};

for a = 1:z

    for k = 1:length(sz)
        d2 = data(data(:,2,a)==sz(k), :, a);
        [mn, idx] = min(d2(:,end));
        rsltrow(k,:) = d2(idx,:);
    end
    disp('Best runs')
    disp(names{a})
    disp(rsltrow)
    
    for k = 2:length(typ)
        figure(a)
        subplot(2,2,k-1)
        dm = data(data(:,1,a)==typ(k),:,a);
        for n = 1:length(tpb)
            dm2 = dm(dm(:,3)==tpb(n),:);
            semilogx(dm2(:,2),dm2(:,4))
            Leg{n} = num2str(tpb(n));
            grid on
            hold on
        end
        legend(Leg)
        title(sprintf('Swap %d per thread',typ(k)));
        xlabel('Problem size')
        ylabel('Time to swap 1e5 times (ms)')
    end
end






