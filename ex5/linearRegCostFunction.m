function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

A = theta'.*X;
b = (y-sum(A,2)).**2;
obsCost = sum(b,1)/(2*m);
regCost= sum(theta(2:end,:).**2,1)*lambda/(2*m);
J=obsCost+regCost;

%Might be something funky with the first 3 lines I think adjustment looks ok;
yHat=sum(A,2);
b = (yHat-y);
tmp=b.*X;
tmp2=(sum(tmp,1)/m)';
adj = theta*lambda/m;
adj(1,1)=0;
grad=tmp2+adj;














% =========================================================================

grad = grad(:);

end
