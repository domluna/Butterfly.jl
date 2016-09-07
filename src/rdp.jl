using ReverseDiffPrototype
const RDP = ReverseDiffPrototype
using Distributions

# 784 * 20 * 10
W1 = randn(784, 20)
W2 = randn(20, 10)
X = randn(32, 784)
Y = rand(DiscreteUniform(1, 10), 32)

# activations
RDP.@forward sigmoid(x) = 1. ./ (1. + exp(-x))
relu(x) = max(0, x)
function elu(x; alpha=1.0)
  pos = relu(x)
  neg = (x - abs(x)) * 0.5
  return pos + alpha * (exp(neg) - 1)
end

function softmax(x)
  exped = exp(x)
  return exped ./ sum(exped, 2)
end

function ce_loss(x, y)
  inds = zip(1:length(y), y)
  return -mean(log(map(i -> x[i[1], i[2]], inds)))
end

function neural_net(w1, w2, x, y)
  x2 = elu(x * w1)
  logits = elu(x2 * w2)
  return ce_loss(softmax(logits), y)
end

function nn_backward(w1, w2, x, y)
  dw1, dw2 = RDP.gradient((w1, w2) -> neural_net(w1, w2, x, y), (w1, w2))
  return dw1, dw2
end

nn_backward(W1, W2, X, Y)
gc()
@time nn_backward(W1, W2, X, Y)
