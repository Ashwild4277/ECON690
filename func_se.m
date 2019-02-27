function [se,t_stat] = func_se(theta_hat,X)
sample = size(X,1);
size_theta = size(theta_hat,1);

beta_hat = theta_hat(1:end-1);
sigma_hat = theta_hat(end);

%The following equations are from textbook page677.
gamma = beta_hat /sigma_hat; 
xg = X*gamma;
phi = normpdf(xg);
PHI = normcdf(xg);
rate=(phi.^2)./(1-PHI); 
% above is to make the following calculations easier.

a = - (xg.*phi-rate-PHI)/(sigma_hat^2);
b = ((xg.^2).*phi + phi - (xg.*rate))/(2*sigma_hat^3);
c = -((xg.^3).*phi + xg.*phi-(xg.*rate) - 2*PHI)/(4*sigma_hat^4);

A_sum = zeros(size_theta);
for i=1:sample
    A = [a(i)*X(i,:)'*X(i,:),b(i)*(X(i,:)'); b(i)*X(i,:),c(i)];
    A_sum = A_sum + A;
end
%compute sum of A
Avar = inv(A_sum);
% compute the estimated asymptotic variance of theta hat.
se=sqrt(diag(Avar));
% the asymptotic standard errors are the square roots of the diagonal
% elements of the \hat{Avar(\hat{\theta})} (textbook page 480)
t_stat=beta_hat./se(1:end-1);
end

