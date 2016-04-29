clear
clc
close all

x1 = 2:2:32;
x = fliplr(x1);
ht = length(x1);
liner = 0:1:271;
sta = 1;
rght = zeros(16,2);
lft = rght;
for k = 1:ht
  stp = (sta-1)+x(k);
  tm = liner(sta:stp);
  sta = stp+1;
  lft(k,:) = tm(1:2);
  rght(k,:) = tm(end-1:end);
end
fprintf('LEFT:')
disp(lft)
fprintf('RIGHT:')
disp(rght)
dlmwrite('CriticalCudaIndexes.txt',[lft, rght])