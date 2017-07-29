clf;
clc;

df = load('shuttle.dat');

X = df(1:end-10000, 1:end-1);
Y = df(1:end-10000, end);

% X = df(:,1:end-1);
% Y = df(:,end);

testX = df(end-10000:end, 1:end-1);
testY = df(end-10000:end,end);

clear df;

% Parameter for input space selection
% Please type >> help fsoperations; to get more information  

k = 6;
function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'ds'; 

% Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,...
    window,testX,testY);
