function [LL] = loglikelyhood(Y,X,theta)
beta = theta(1:(end-1));
sigma = theta(end);
LL = -mean((Y==0).*(log(normcdf((-X*beta)./sigma)))+(Y~=0).*(log(normpdf((Y-X*beta)./sigma))-log(sigma)));
end