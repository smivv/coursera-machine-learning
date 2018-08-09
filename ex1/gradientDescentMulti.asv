function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %
    
    sum_0 = 0;
    sum_1 = 0;
    for i = 1:length(X)
        v = X(i, :)*theta - y(i);
        sum_0 = sum_0 + v*X(i, 1);
        sum_1 = sum_1 + v*X(i, 2);
    end

    theta_0 = theta(1) - alpha * sum_0/m;
    theta_1 = theta(2) - alpha * sum_1/m;

    theta = [theta_0; theta_1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
