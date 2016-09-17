using ReverseDiffPrototype
const RDP = ReverseDiffPrototype
using Distributions

include("mnist_loader.jl")

W1 = randn(Float32, 784, 20) * 0.01
W2 = randn(Float32, 20, 10) * 0.01

# Activations
# These go from n -> n so I think doing forward diff
# is beneficial though I'm not 100% sure.
RDP.@forward sigmoid(x) = 1. ./ (1. + exp(-x))
RDP.@forward relu(x) = max(0., x)

function softmax(logits)
    exp_logits = exp(logits)
    return exp_logits ./ sum(exp_logits, 2)
end

logsoftmax(logits) = log(softmax(logits))

function cross_entropy_loss(logprobs, y)
    inds = zip(1:length(y), y)
    return -mean(map(i -> logprobs[i[1], i[2]], inds))
end

function neural_net(w1, w2, x)
    x2 = sigmoid(x * w1)
    return x2 * w2
end

function train_neural_net(w1, w2, x, y)
    logits = neural_net(w1, w2, x)
    return cross_entropy_loss(logsoftmax(logits), y)
end

function predict_neural_net(w1, w2, x)
    logits = neural_net(w1, w2, x)
    preds = Int[]
    for i=1:size(logits,1)
        push!(preds, indmax(logits[i, :]))
    end
    return preds
end

function nn_backward(w1, w2, x, y)
    dW1, dW2, dX = RDP.gradient((w1, w2, x) -> train_neural_net(w1, w2, x, y), (w1, w2, x))
    return (dW1, dW2)
end

accuracy(ypred, ytest) = mean(ypred .== ytest)

α = 1e-3
loader = DataLoader()
x, y = next_batch(loader, 128)

# warmup
dW1, dW2 = nn_backward(W1, W2, x, y)
gc()

testx, testy = load_test_set() 

for i in 1:10000
    x, y = next_batch(loader, 128)
    dW1, dW2 = nn_backward(W1, W2, x, y)
    # Vanilla SGD
    W1 += α * dW1
    W2 += α * dW2

    if i % 1000 == 0
        ypred = predict_neural_net(W1, W2, testx)
        acc = accuracy(ypred, testy)
        println("Loop Iter: $(i), Test Accuracy = $(acc)")
    end
end