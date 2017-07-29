% data avaialable before simulating
A = (-10:0.05:10)';
Z = cos(A) + cos(2*A) + 0.1.*rand(size(A));

order = 10;
% Re-arrange the data points into a Hankel matrix for (N)AR time-series modeling
X = windowize(Z,1:(order+1));
Y = X(:,end);
X = X(:,1:order);

% train the model
gam = 10; sig2 = 10;
[alpha, b] = trainlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'});
% make plot the model
figure(1);clf;
plotlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b});

% optimize the lssvm using crossvalidate
Xnew = Z((end-order+1):end)';
Z(end+1) = simlssvm({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, {alpha, b}, Xnew);

% split data into training an validation set
test_size = 100;
Ztrain = Z(1:length(Z) - test_size);
Ztest = Z(length(Z) - test_size+1:end);

% make a prediction
horizon = length(Ztest) - order;
Zhat = predict({X,Y,'f',gam,sig2,'RBF_kernel', 'preprocess'}, Ztest(1:order),horizon);

Ztest_w = Ztest(order+1:end);

figure(2);clf;
plot([Ztest_w Zhat]);

mse = sum(power((Ztest_w - Zhat),2)) * (1 / length(Zhat));
disp('mean square error:');
disp(mse);
