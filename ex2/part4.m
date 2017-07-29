clc;
clear all;
close all;
% In situations where the data is corrupted with non-Gaussian noise or outliers, it becomes important to incorporate
% robustness into the estimation. Consider the following simple example:
X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));
% Outliers are added via:
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));
%%
% Train and plot an LS-SVM regression model with gam = 100, sig2 = 0.1. What is the influence of
% the outlying data points?
gam = 100;sig2 = 0.1;
[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
figure;hold on;
plotlssvm({X,Y,'f',gam,sig2},{alpha,b});

%%
% Let’s now train a robust LS-SVM model using the object-oriented interface and robust crossvalidation.
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'whuber';
close all;
for j=1:10
model = tunelssvm(model,'simplex',costFun,{10,'mse'},wFun);
model = robustlssvm(model);

figure(j);clf;hold on;
plotlssvm(model);
end

%%
%check out the different cost functions
for j=1:4
    model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
    costFun = 'rcrossvalidatelssvm';
    % possilibities : [whuber|whampel|wlogistic|wmyriad]
    switch j
           case 1
            wFun = 'whuber';
        case 2
            wFun = 'whampel';
        case 3
            wFun = 'wlogistic';
        case 4
            wFun = 'wmyriad';
    end
    model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
    model = robustlssvm(model);
    figure(j);clf;hold on;
    plotlssvm(model);
end

% | from the manual:
% | -> In the robust cross-validation case, other possibilities for the weights are whampel, wlogistic and
% | -> wmyriad.