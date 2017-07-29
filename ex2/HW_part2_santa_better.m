clear all;
close all
load santafe;
%%
order    = 50;
X        = windowize(Z,1:order+1);
Xtrain   = X(1:end-order,1:order);
Ytrain   = X(1:end-order,end); 
Xtest       = Z(end-order+1:end,1);
% 
% [gam,sig2,cost] = tunelssvm({Xtrain,Ytrain,'f',[],[],'RBF_kernel','csa','original'},  ...
%  'simplex','crossvalidatelssvm',{100,'mae'});
gam=10000;
sig2=20000;


% bayens This result is worse
% [~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
% [~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
% [~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

% ---------------------
% robust takes WAAAy to long, maybe get this faster somehow?
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
model = tunelssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','csa','original'},'simplex',costFun,{10,'mae'},wFun);
model = robustlssvm(model);
% ---------------------

% [alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','csa','original'});

figure(1);plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','csa','original'}, {alpha, b});

prediction = predict({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel','csa','original'},Xtest,200);
figure(2);plot([prediction Ztest]);

mse = sum(power((Ztest - prediction),2)) * (1 / length(Ztest));
disp(mse);
