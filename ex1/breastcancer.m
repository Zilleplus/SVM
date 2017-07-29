%% Linear Case

load breast;

Xtrain  = trainset;
Ytrain  = labels_train;
Xtest = testset;
Ytest = labels_test;

type='c'; 
model = {Xtrain,Ytrain,type,[],[],'lin_kernel','ds'};
gam = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'});

[Ysim, Ylatent] = simlssvm({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, {alpha,b}, Xtest);

error_test = sum(Ysim~=Ytest); 
fprintf('\n On Test Data: N of Misclassifications = %d, error rate = %.2f%%', error_test, error_test/length(Ytest)*100)

performance = crossvalidate({Xtrain,Ytrain,type,gam,[],'lin_kernel'}, 10,'misclass');
fprintf('\n Crossvalidation: error rate = %.2f%%', performance*100);

roc(Ylatent, Ytest);

%% RBF Case

load breast;

Xtrain  = trainset;
Ytrain  = labels_train;
Xtest = testset;
Ytest = labels_test;

type='c'; 
model = {Xtrain,Ytrain,type,[],[],'RBF_kernel','ds'};
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm',{10,'misclass'});

[alpha,b] = trainlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'});

[Ysim, Ylatent] = simlssvm({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, {alpha,b}, Xtest);

error_test = sum(Ysim~=Ytest); 
fprintf('\n On Test Data: N of Misclassifications = %d, error rate = %.2f%%', error_test, error_test/length(Ytest)*100)

performance = crossvalidate({Xtrain,Ytrain,type,gam,sig2,'RBF_kernel'}, 10,'misclass');
fprintf('\n Crossvalidation: error rate = %.2f%%', performance*100);

roc(Ylatent, Ytest)