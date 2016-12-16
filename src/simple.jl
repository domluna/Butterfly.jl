using ReverseDiff
using Distributions

include("mnist_loader.jl")

W1 = randn(Float32, 784, 128) * 0.1 
W2 = randn(Float32, 128, 10) * 0.1

ReverseDiff.@forward sigmoid(x) = 1. ./ (1. + exp(-x))

function softmax(logits)
    exp_logits = exp(logits)
    return exp_logits ./ sum(exp_logits, 2)
end

logsoftmax(logits) = log(softmax(logits))

function cross_entropy_loss(logprobs, y)
    loss = 0
    batch_size = size(logprobs, 1)
    for i = 1:batch_size
        y_idx = Int(y[i])
        loss += logprobs[i, y_idx]
    end
    return -loss/batch_size
end

function neural_net(w1, w2, x, y)
    x2 = sigmoid(x * w1)
    logits = x2 * w2
    return cross_entropy_loss(logsoftmax(logits), y)
end

function eval_neural_net(w1, w2, x)
    x2 = sigmoid(x * w1)
    logits = x2 * w2
    preds = Int[]
    for i=1:size(logits,1)
        push!(preds, indmax(logits[i, :]))
    end
    return preds
end

accuracy(ypred, ytest) = mean(ypred .== ytest)

α = 1e-2
loader = DataLoader()
x, y = next_batch(loader, 100)
xs = (W1, W2, x, y)
outs = map(similar, xs)
const f! = ReverseDiff.compile_gradient(neural_net, xs)
@time f!(outs, (W1, W2, x, y))

testx, testy = load_test_set()

for i=1:1000
  x, y = next_batch(loader, 100)
  # Vanilla SGD
  f!(outs, (W1, W2, x, y))
  W1 -= α * outs[1]
  W2 -= α * outs[2]

  if i % 100 == 0
      preds = eval_neural_net(W1, W2, testx)
      acc = accuracy(preds, testy)
      println("Loop Iter: $(i), Test Accuracy = $(acc)")
  end
end
