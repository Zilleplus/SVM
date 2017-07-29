% close all;

X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
% The training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));
%%
% The following commands can be used to make an LS-SVM model with the RBF kernel and arbitrary values
% of the hyper-parameters:
gam = 10; % allow errors
sig2 = 10^-6; % make it flexibel
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});
% The results on the training set can be visualized via:
figure(1);clf;hold on;
plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});
% The results of applying the trained model on the test set are calculated using:
YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},...
{alpha,b},Xtest);
% and visualized with:
figure(2);clf;hold on;
plot(Xtest,Ytest,'.');
hold on;
plot(Xtest,YtestEst,'r+');
legend('Ytest','YtestEst');
% Make similar plots showing the results on training and test sets with different values of gam and sig2. Do
% you think there is one pair of optimal hyper-parameters?
