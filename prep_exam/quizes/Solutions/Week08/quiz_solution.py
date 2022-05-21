import math

x = 120

# hidden layer with ReLu
h1 = max(10 - 0.01*x, 0)
h2 = max(0.1*x, 0)

# output layer
y1_hat = 4 + 0.05*h1
y2_hat = -0.01*h1 + 0.2*h2

# softmax
y1 = math.exp(y1_hat)/(math.exp(y1_hat)+math.exp(y2_hat))
y2 = math.exp(y2_hat)/(math.exp(y1_hat)+math.exp(y2_hat))

# loss
L = -math.log(y2) # since we know it belongs to class 2

# partial derivative, we know that t2 = 1
dLdw22 = h2 * (y2-1)
