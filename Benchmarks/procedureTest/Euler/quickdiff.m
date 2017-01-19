clear
clc
close all

af = @(v,x,t) 6*sin(t)-x^5-.1*v;
tf = 50;
dt = .001;
ta = 0:dt:tf;
v = zeros(1,length(ta));
x = v;
tic
x(1) = 3;
v(1) = 0;
a = v;
a(1) = af(v(1),x(1),ta(1));
for k=2:length(ta)
    v(k) = v(k-1)+dt*a(k-1);
    x(k) = x(k-1)+dt*v(k-1);
    a(k) = af(v(k),x(k),ta(k));
end
toc
plot(ta,x)
hold on
g = input('Press ENTER to see velocity');
plot(ta,v)
g = input('Press ENTER to see acceleration');
grid on
plot(ta,a)
legend('x','v','a')


