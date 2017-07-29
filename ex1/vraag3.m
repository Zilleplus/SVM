clc;
clear;
load iris;

% set the parameters to some value
gam = 0.1;
sig2 = 20;

% generate random indices
idx = randperm(size(X,1));
% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

%%
% Calculate the performance in terms of misclassification error on the validation set (using estYval
% and Yval). Make a plot of the performance with respect to several values of gam and sig2 (e.g.
% gam= 1, 10, 100, sig2= 0.1, 1, 10). Give comments about the results.
gam_values = [ 1 10 100 1000 ];
sig2_values = [ 0.1 1 10 100 1000];

error_rate_gam_sig2 =  zeros(length(gam_values),length(sig2_values));

for gam_i=1:length(gam_values)
    for sig2_i=1:length(sig2_values)
        % train the model
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam_values(gam_i),sig2_values(sig2_i),'RBF_kernel'});
        % evaluate at Xval
        estYval = simlssvm({Xtrain,Ytrain,'c',gam_values(gam_i),sig2_values(sig2_i),'RBF_kernel'},{alpha,b},Xval);
        % test the performace
        error_rate_gam_sig2(gam_i,sig2_i) = sum(estYval~=Yt);
    end
end

error_rate_gam_sig2 = error_rate_gam_sig2./length(estYval);

% plot for different gamma values
figure(1);
clf;
for gam_i=1:length(gam_values)
    subplot(2,2,gam_i);
    plot(sig2_values,error_rate_gam_sig2(gam_i,:));hold on;
    
    titlename =strcat( 'gamma=',num2str(gam_values(gam_i)) );
    title( titlename );
    xlabel('sigma2');
    ylabel('error rate');
end

% plot for different sigma2 values
figure(2);
clf;
for sigma2_i=1:length(gam_values)
    subplot(2,2,sigma2_i);
    plot(gam_values,error_rate_gam_sig2(:,sigma2_i));hold on;
    
    titlename =strcat( 'sigma2=',num2str(sig2_values(sigma2_i)) );
    title( titlename );
    xlabel('gamma');
    ylabel('error rate');
end

%% Perform crossvalidation using 10 folds:
gamma =10;
sigma2 =5;

% train the model
% [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',sigma2,gamma,'RBF_kernel'});
% % evaluate at Xval
% estYval = simlssvm({Xtrain,Ytrain,'c',gamma,sigma2,'RBF_kernel'},{alpha,b},Xval);

% Perform crossvalidation using 10 folds:

for gamma=1:10
    gamma
    for sigma2=1:10
        performance_cross_10_fold = crossvalidate({X,Y,'c',gamma,sigma2,'RBF_kernel'},10,'misclass');
        performance_one_out = leaveoneout({X,Y,'c',gamma,sigma2,'RBF_kernel'},'misclass');
        out(gamma,sigma2) = abs(performance_cross_10_fold - performance_one_out);
    end
end

performance_cross_10_fold = crossvalidate({X,Y,'c',gam,sig2,'RBF_kernel'},10,'misclass')
performance_one_out = leaveoneout({X,Y,'c',gam,sig2,'RBF_kernel'},'misclass')

% Think about a clear and intuitive way to represent this technique. Why should one prefer this method
% over a simple validation? Change crossvalidate procedure for leaveoneout (removing the
% 10). Is it giving better results? In which cases one would prefer each?

%%
% 3. Use tunelssvm procedure to optimize the hyperparameters. Execute:
model = {X,Y,'c',[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'simplex','crossvalidatelssvm',{10,'misclass'});
cost
% Try to change different parameters like 'csa' (Coupled Simulated Annealing) vs. 'ds' (Randomized
% Directional Search) and 'simplex' (Nelder-Mead method) vs. 'gridsearch' (brute force
% gridsearch). What differences do you observe? Why in some cases the obtained hyperparameters differ
% a lot?

%%
% 4. One alternative way to judge a classifier is by using the Receiver Operating Characteristic (ROC)
% curve of a binary classifier (see >> help roc). The higher the area under the curve, the better the
% classifier separates the data. For making the ROC curve on the training data use roc command. Using
% the training set is not recommended, do you know why? A ROC plot of a validation set Xval and
% Yval is made by

[alpha,b] = trainlssvm({X,Y,'c',gam,sig2,'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({X,Y,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
roc(Ylatent,Yval);