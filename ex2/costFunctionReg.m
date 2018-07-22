function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
A = theta'.*X; 
yHat = sigmoid(sum(A,2));
obsCost = -y.*log(yHat)-(1.-y).*(log(1.-yHat));
regCost= sum(theta(2:end,:).**2,1)*lambda/(2*m);
J=sum(obsCost,1)/m + regCost;

%Get the gradient;
tmp=(yHat-y).*X;
tmp2=(sum(tmp,1)/m)';
adj = theta*lambda/m;
adj(1,1)=0;
grad=tmp2+adj;






% =============================================================

end
