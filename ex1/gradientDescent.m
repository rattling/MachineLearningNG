function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    %First need to compute y hat vector
    %so first i need to form the matrix theta x
    A=theta'.*X;
    b=sum(A,2);
    %disp(A);
    %so b contains the estimates a single col vector

    %then get the difference with y;
    c=b-y;
    %now can multiply it element wise by x;
    d=c.*X;
    %now can sum over m;
    e=sum(d,1);
    %now can multiply by alpha/m;
    f=alpha/m*e;
    %new thete is old theta - f;
    theta=theta-f';
    






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
