%% Question: 
% This question asks you to use MATLAB to estimate a Tobit model and
% interpret your results as in Wooldrige problem 17.5 parts a-c.
%% Import data
% First export dta file to csv using stata.
% Then inport csv file
fringe = readtable("D:\MATLAB\ECON690\PS3\fringe.csv");
%% 1. Examine the data in MATLAB. What is the sample size?
sample = size(fringe,1);
% (note: 1:count the row of the matrix; 2:column of the matrix)
% sample size = 616
%% 2. Estimate a linear model by OLS relating hrbens (hourly benefits) to exper, age, educ,tenure, married, male, white, nrtheast, nrthcen, south,and union.
Y = table2array(fringe(:,"hrbens"));
X = table2array(fringe(:,["exper", "age", "educ", "tenure", "married",...
    "male", "white", "nrtheast","nrthcen", "south", "union"]));
X = [ones(616,1) X];%including the constant term
beta_OLS = OLS(X,Y);
%% 3. Now estimate a Tobit model of hourly benefits, using the same explanatory variables as in part c. Compare the results to those in part(b).
size_theta = size(X,2)+1; %size of theta
theta_0 = zeros(size_theta, 1); %set an initial value

A = [];
b = [];
Aeq = [];
beq = [];
lb= [-inf* ones(size_theta-1,1);0]; %sigma>0
ub =inf* ones(size_theta,1); 

fun = @(theta) loglikelyhood(Y,X,theta);
theta_hat = fmincon(fun,theta_0,A,b,Aeq,beq,lb,ub);

beta_hat = theta_hat(1:end-1);
sigma_hat = theta_hat(end);
clear A Aeq b beq lb ub

%%
% the standard error:
%The equations are on textbook page677.
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
    A_sum = A_sum+A;
end
%compute sum of A

Avar = inv(A_sum);
% compute the estimated asymptotic variance of theta hat.
se=sqrt(diag(Avar));
% the asymptotic standard errors are the square roots of the diagonal
% elements of the \hat{Avar(\hat{\theta})} (textbook page 480)
t_stat=beta_hat./se(1:end-1);

clear a b c gamma xg phi PHI rate Avar A_sum

%{ 
reason:
sum (Y>0)=575
sum (Y==0)=41
The results are similar because very few observations are censored under
the criteria of Y==0. The Tobit estimator is larger because some
observations are censored
%}
%%
%{
Recheck this later_1: 
A = [zeros(1,size-1),-1];
fun = @(theta) loglikelyhood(Y_censored,X,theta);
theta_hat = fmincon(fun,theta,A,-1); 
doesn't work
Why?
%}
%{
Recheck this later_2: 
What if we 
change loglikelyhood(Y,X,theta) to loglikelyhood(theta); 
and change 
    fun = @(theta) loglikelyhood(Y,X,theta);
    theta_hat = fmincon(fun,theta_0,A,b,Aeq,beq,lb,ub);
to 
    theta_hat = fmincon(loglikelyhood,theta_0,A,b,Aeq,beq,lb,ub);
%}
%Recheck this later_3:why func_se(X,theta_hat); doesn't work?

%% 4. Add the square of both experience and tenure to the Tobit model from part c. Do you think these should be included? Write code to compute the standard error of the estimated coefficients and explain what formula(s) you use.
new=table2array(fringe(:,["expersq", "tenuresq"]));
X_new = [X,new];
size_new = size(X_new,2)+1; %size of theta
theta_0 = zeros(size_new, 1); %set an initial value

A = [];
b = [];
Aeq = [];
beq = [];
lb= [-inf* ones(size_new-1,1);0]; %sigma>0
ub =inf* ones(size_new,1); 

fun = @(theta) loglikelyhood(Y,X_new,theta);
theta_hat_new = fmincon(fun,theta_0,A,b,Aeq,beq,lb,ub);

beta_hat_new = theta_hat_new(1:end-1);
sigma_hat_new = theta_hat_new(end);
clear A Aeq b beq lb ub

%%
% the standard error:
%The equations are on textbook page677.
gamma = beta_hat_new /sigma_hat_new; 
xg = X_new*gamma;
phi = normpdf(xg);
PHI = normcdf(xg);
rate=(phi.^2)./(1-PHI); 
% above is to make the following calculations easier.

a = - (xg.*phi-rate-PHI)/(sigma_hat_new^2);
b = ((xg.^2).*phi + phi - (xg.*rate))/(2*sigma_hat_new^3);
c = -((xg.^3).*phi + xg.*phi-(xg.*rate) - 2*PHI)/(4*sigma_hat_new^4);


A_sum = zeros(size_new);
for i=1:sample
    A = [a(i)*X_new(i,:)'*X_new(i,:),b(i)*(X_new(i,:)'); b(i)*X_new(i,:),c(i)];
    A_sum = A_sum+A;
end

%compute sum of A
Avar = inv(A_sum);
% compute the estimated asymptotic variance of theta hat.
se_new=sqrt(diag(Avar));
% the asymptotic standard errors are the square roots of the diagonal
% elements of the \hat{Avar(\hat{\theta})} (textbook page 480)
t_stat_new=beta_hat_new./se_new(1:end-1);

%Both squared terms should be included because they are statistically
%significant.
