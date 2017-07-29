clear
% close all
X = 3.*randn(100,2);
ssize = 10;
sig2_list=[10000 1000 100 10 1 0.1 0.01 0.001 0.0001 0.00001]

for k=1:length(sig2_list)
    k
    sig2 = sig2_list(k);
    subset = zeros(ssize,2);
    fig = figure(k);clf;
    for t = 1:100,
      % display progress:
      if(mod(t,10) == 0)
          disp(t);
      end

      %
      % new candidate subset
      %
      r = ceil(rand*ssize);
      candidate = [subset([1:r-1 r+1:end],:); X(t,:)];

      %
      % is this candidate better than the previous?
      %
      if kentropy(candidate, 'RBF_kernel',sig2)>...
            kentropy(subset, 'RBF_kernel',sig2),
        subset = candidate;
      end

      %
      % make a figure
      %
      plot(X(:,1),X(:,2),'b*'); hold on;
      plot(subset(:,1),subset(:,2),'ro','linewidth',6); hold off; 
%       pause(1)
      
    end
    title(['sgima2=' num2str(sig2)]);
%     name = ['sigma_' num2str(sig2)];
%     saveas(fig,name,'png');
end