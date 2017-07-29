load iris;
%%
% Try out the linear kernel with the command
gam=20;
[alpha,b] = trainlssvm({X,Y,'c',gam,[],'lin_kernel'});
figure;plotlssvm({X,Y,'c',gam,[],'lin_kernel'},{alpha,b});


%%
% Try out the polynomial kernel with degree
% parameters
t=1;
gam=1;
type='c';
error_rate =  zeros(8,1);
figure(1);
for i=1:20
    degree = i;
    figure(i);
    clf;
    % train the system
    [alpha,b] = trainlssvm({X,Y,type,gam,[t;degree],'poly_kernel'});
    % test the performace
    % -> evaluate the test values
    [Yht, Zt] = simlssvm({X,Y,type,gam,[t; degree],'poly_kernel'}, {alpha,b}, Xt);
    % -> check if there valid
    error_rate(i) = sum(Yht~=Yt)/ length(Yht);
    % now plot everything
    plotlssvm({X,Y,type,gam,[t; degree],'poly_kernel','preprocess'},{alpha,b});
end
error_rate
%%
% Let us focus on the RBF kernel with kernel parameter the bandwidth
% sig2(?^2)
% 
gam =1;
type='c';
sig2=[0.01 0.1 0.2 0.5 1 2 5 10 20];
error_rate =  zeros(length(sig2),1);

% train the system
for i=1:length(sig2)
    [alpha,b] = trainlssvm({X,Y,type,gam,sig2(i),'RBF_kernel'});
    
    % evalute the test points
    [Yht, Zt] = simlssvm({X,Y,type,gam,sig2(i),'RBF_kernel'}, {alpha,b}, Xt);
    % calculate the errors
    error_rate(i) = sum(Yht~=Yt)/ length(Yht);
     
    % now plot everything
    figure(i);
    plotlssvm({X,Y,type,gam,sig2(i),'RBF_kernel','preprocess'},{alpha,b});
end
figure(length(sig2)+1);
sig2'
error_rate

%%
% Now, take a look at the regularization constant gam. Fix a reasonable choice for the sig2 of the RBF
% kernel and again compare a range of gam’s by plotting the corresponding test set performances. What
% is a good range for gam?

sig2 =1;
type='c';
gam=[   0.01 0.1 0.5 1 5 10 20 100 1000];
error_rate =  zeros(length(gam),1);

% train the system
for i=1:length(gam)
    [alpha,b] = trainlssvm({X,Y,type,gam(i),sig2,'RBF_kernel'});
    
    % evalute the test points
    [Yht, Zt] = simlssvm({X,Y,type,gam(i),sig2,'RBF_kernel'}, {alpha,b}, Xt);
    % calculate the errors
    error_rate(i) = sum(Yht~=Yt)/ length(Yht);
     
    % now plot everything
    figure(i);
    plotlssvm({X,Y,type,gam(i),sig2,'RBF_kernel','preprocess'},{alpha,b});
end
figure(length(sig2)+1);
gam'
error_rate


