close all;
clear all;
clc;

load santafe;
%%
order  = 50;
X      = windowize(Z,1:(order+1));
Y      = X(:,end);
X      = X(:,1:order);

% tune the regressor
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa','original'}, ...
   'simplex', 'crossvalidatelssvm', {10, 'mae'});

% train the regressor
[alpha, b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'});
figure(1);plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'}, {alpha, b});

Zhat = predict({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'}, Ztest,length(Ztest));

figure(2); plot([Ztest Zhat]);

mse = sum(power((Ztest - Zhat),2)) * (1 / length(Zhat));
disp(mse);
