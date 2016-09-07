using ReverseDiffPrototype
const RDP = ReverseDiffPrototype
using Distributions

# 784 * 20 * 10
W1 = randn(784, 20)
W2 = randn(20, 10)
X = randn(32, 784)
Y = rand(DiscreteUniform(1, 10), 32)

RDP.@forward sigmoid(x) = 1. ./ (1. + exp(-x))

function softmax(x)
    xx = x - maximum(x)
    exped = exp(xx)
    return exped ./ sum(exped, 2)
end

function ce_loss(x, y)
    bs = size(x, 1)
    inds = zip(1:bs, y)
    return mean(log(map(i -> x[i[1], i[2]], inds)))
end

function neural_net(w1, w2, x, y)
    x2 = sigmoid(x * w1)
    logits = sigmoid(x2 * w2)
    return ce_loss(softmax(logits), y)
end

function nn_backward(w1, w2, x, y)
    ∇w1, ∇w2, ∇x, ∇y = RDP.gradient(neural_net, (w1, w2, x, y))
    return (∇w1, ∇w2)
end

nn_backward(W1, W2, X, Y)
gc()
@time nn_backward(W1, W2, X, Y)



