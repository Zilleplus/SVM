close all;
clc;
clear all;

X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
% The training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));
%%
% The functions crossvalidate and leaveoneout can be used to estimate the performance of a set of
% hyper-parameters:
gam=10; %start value
sig2=0.01; %start value

% cost_crossval = crossvalidate({Xtrain,Ytrain,'f',gam,sig2},10);
% cost_loo = leaveoneout({Xtrain,Ytrain,'f',gam,sig2});

% The optimization of the hyper-parameters is done with the function tunelssvm. For example,
optFun = 'simplex';% gridsearch or simplex
globalOptFun = 'ds'; % csa or ds
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ...
globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'})
% \-> using cost function crossvalidatelssvm
%%
% tunes the hyper-parameters using 10-fold crossvalidation using a gridsearch approach and Coupled Simulated
% Annealing. Train a model with tuned hyper-parameters and display the results on the training data:
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
figure;hold on;
plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});
% Run the previous tunelssvm command several times. What can you say about the values of the hyper-
% parameters and the results? Change optFun to 'simplex' (>> optFun = 'simplex';) and run
% the tuning once more. What is the difference between 'simplex' and 'gridsearch'? Try to change
% globalOptFun to 'ds' (Randomized Directional Search). Is there any significant deviation in results?
% Which method is faster and why

%%
N=100;
globalOptFun = 'csa'; % csa or ds

sigm2_smp =1:N;
gam_smp =1:N;
cost_smp=1:N;

sigm2_grid =1:N;
gam_grid =1:N;
cost_grid =1:N;
for i=1:N    
    [sigm2_smp(i),gam_smp(i),cost_smp(i)] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ...
    globalOptFun},'simplex','crossvalidatelssvm',{10,'mse'});

    [sigm2_grid(i),gam_grid(i),cost_grid(i)] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ...
    globalOptFun},'gridsearch','crossvalidatelssvm',{10,'mse'});
end

min(cost_smp)
max(cost_smp)

min(cost_grid)
max(cost_grid)




