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

sig2 = 0.5; gam = 10;
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)
% The model can be optimized with respect to these criteria:
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);
%%
% Try to think about a clear schematic visualization of this three-level principle.
% For regression, the error bars can be computed using Bayesian inference, e.g.
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');
% For classification, it is also possible to get probability estimates. Load the Iris data-set:
%%
clear;
load iris;
gam = 5; sig2 = 1;
% The probabilities that a certain data point belongs to the positive class given the model is calculated by:
bay_modoutClass({X,Y,'c',gam,sig2},'figure');
% How do you interpret the colors? (hint: activate the color-bar of the figure). Change the values of gam and
% sig2. What is the influence of the given hyper-parameters on this figure?
%%
% The Bayesian framework can also be used to select the most relevant inputs by Automatic Relevance Deter-
% mination (ARD). The following procedure uses this criterion for backward selection for a three dimensional
% input selection task:
X = 10. * rand(100,3)-3;
Y = cos(X(:,1)) + cos(2 * (X(:,1))) + 0.3.* randn(100,1);
[selected, ranking] = bay_lssvmARD({X,Y,'class',gam,sig2});