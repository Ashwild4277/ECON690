function [beta_OLS] = OLS(X,Y)
%beta_OLS = (X'X)^(-1)X'Y
beta_OLS = (transpose(X)*X)\(transpose(X)*Y) ;
end