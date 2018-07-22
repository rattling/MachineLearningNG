function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(size(X, 1), 1) X];
z2=a1*Theta1';
a2=sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2];
z3=a2*Theta2';
a3=sigmoid(z3);
I = eye(max(y));
y01 = I(y,:);
obsCost = -y01.*log(a3)-(1.-y01).*(log(1.-a3));
square1=Theta1(:,2:end).^2;
totSquare1=sum(square1(:));
square2=Theta2(:,2:end).^2;
totSquare2=sum(square2(:));
reg=(totSquare1+totSquare2)*lambda/(2*m);
%reg=0;
J=(sum(obsCost(:))/m) +reg;
%THERE IS NO DEPENDENCY BETWEEN INPUT RECORDS SO EASIER TO DO IT AS VECTORIZED SOLUTION;
%MAIN THING IS CHECKING THE VECTOR SIZE AND SHAPE AS I GO TO MAKE SURE I HAVE THE RIGHT MATRIX;

%Calculate the output error;
outputError=a3-y01;
%Calculate the Hidden Layer error;
%I think i need to lop the first or lats row off of theta2 as we dont care about bias
%maybe it is better to do this for one input a time aswell so can see what its going
hiddenError=(Theta2(:,2:end)'*outputError')'.*sigmoidGradient(z2);
x=size(hiddenError);
delta2=outputError'*a2;
delta1=hiddenError'*a1;
Theta1_grad=delta1./m
Theta2_grad=delta2./m

Theta1(:,1) = 0 ; 
Theta2(:,1) = 0 ; 
Theta1_grad = Theta1_grad + Theta1*lambda/m;
Theta2_grad = Theta2_grad + Theta2*lambda/m;

d2=size(delta2);
d1=size(delta1);



% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
