clc;
clear;
load('ripley.mat');

% create the validation data

% - generate random indices
idx = randperm(size(X,1));
% - create the training and validation sets
% - using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

%%
% display the data
figure;
scatter(X(1:125,1),X(1:125,2)); hold on;
scatter(X(126:250,1),X(126:250,2));
%%
type = 'c';
% optimize the parameter gam
gam = tunelssvm({X,Y,type,[],[],'lin_kernel','csa'}, 'simplex','crossvalidatelssvm',{10,'misclass'})
% create the linear model
[alpha,b] = trainlssvm({X,Y,type,gam,[],'lin_kernel'});
% plot the model to get an idea of how good/bad it is
figure;plotlssvm({X,Y,type,gam,[],'lin_kernel'},{alpha,b});

% calculate the performance 10 fold cross
performance_cross_10_fold = crossvalidate({X,Y,'c',gam,[],'lin_kernel'},20,'misclass')

% ROC curve lin model
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,[],'lin_kernel'},{alpha,b},Xval);
roc(Ylatent,Yval);

% finally test the performance of the model
[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
error_rate_lin = sum(Yht~=Yt)/ length(Yht)
%%
model = {X,Y,'c',[],[],'RBF_kernel','csa'};
% optimize the model
[gam, sig2, cost]  = tunelssvm(model,'simplex','crossvalidatelssvm',{20,'misclass'});
% train the model
type = 'c';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});
% plot the model to get an idea of how good/bad it is
figure;plotlssvm({X,Y,type,gam,sig2,'RBF_kernel'},{alpha,b});

% calculate the formance 10 fold cross
performance_cross_10_fold = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass')

% ROC curve lin model
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
roc(Ylatent,Yval);

% evalute the test points
estYt = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xt);
% get the error rate on the test data
error_rate = sum(estYt~=Yt)/ double(length(Yt))
