using ReverseDiffPrototype
const RDP = ReverseDiffPrototype
using Distributions

include("mnist_loader.jl")

W1 = randn(Float32, 784, 10) * 0.01

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

function neural_net(w1, x)
    return sigmoid(x * w1)
end

function train_neural_net(w1, x, y)
    logits = neural_net(w1, x)
    return cross_entropy_loss(logsoftmax(logits), y)
end

function predict_neural_net(w1, x)
    logits = neural_net(w1, x)
    preds = Int[]
    for i=1:size(logits,1)
        push!(preds, indmax(logits[i, :]))
    end
    return preds
end

function nn_backward(w1, x, y)
    dW1, dX = RDP.gradient((w1, x) -> train_neural_net(w1, x, y), (w1, x))
    return dW1
end

accuracy(ypred, ytest) = mean(ypred .== ytest)

α = 1e-3
loader = DataLoader()
x, y = next_batch(loader, 128)

# warmup, see if grads make sense
dW1 = nn_backward(W1, x, y)
println(dW1[1, 1:5])

testx, testy = load_test_set() 

for i in 1:10000
    x, y = next_batch(loader, 128)
    dW1 = nn_backward(W1, x, y)
    # Vanilla SGD
    W1 += α * dW1

    if i % 1000 == 0
        ypred = predict_neural_net(W1, testx)
        acc = accuracy(ypred, testy)
        println("Loop iter $(i), accuracy = $(acc)")
    end
end