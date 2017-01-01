using ReverseDiff
using Distributions
include("char_loader.jl")

loader = CharLoader("../data/input.txt")

# number of unique chars in the dataset
BATCH_SIZE = 25
INPUT_SIZE = length(loader.chars)
OUTPUT_SIZE = length(loader.chars)
STATE_SIZE = 100

Wh = randn(T, state_size, state_size) * 0.01
Wx = randn(T, state_size, input_size) * 0.01
Wo = randn(T, output_size, state_size) * 0.01

ReverseDiff.@forward minus_log(x) = -log(x)
softmax(x) = (exp_x = exp.(x); exp_x ./ sum(exp_x, 1))
cross_entropy_loss(x, y) = mean(sum(y .* minus_log.(x), 1))

function train_rnn(cell::AbstractCell{T}, X, Y)
    loss = T(0)
    n = size(X, 2)
    ht = zeros(1, state_size)
    for t=1:n
        ot, ht = rnn_cell(Wx, Wh, Wo, X[t, :], ht)
        y = Y[t]
        loss += minus_log.(softmax(ot))[y]
    end
    return loss / n
end

function predict_neural_net(Wx, Wh, Wo, X)
    preds = Int[]
    T = size(X, 1)
    ht = zeros(1, state_size)
    for t=1:T
        xt = view(X, t, :)
        ot, ht = rnn_cell(Wx, Wh, Wo, xt, ht)
        push!(preds, indmax(ot))
    end
    return preds
end

function generate_text(T)
    id0 = rand(1:input_size, 1)[1]
    xt = zeros(1, input_size)
    xt[id0] = 1
    ht = zeros(1, state_size)
    ids = Int[id0]

    for t=1:T
        ot, ht = rnn_cell(Wx, Wh, Wo, xt, ht)
        probs = softmax(ot)
        ct = Categorical(probs[:])
        pred = rand(ct)
        xt[:] = 0.
        xt[pred] = 1.
        push!(ids, pred)
    end
    return ids
end

function train_model(iters, Wx, Wh, Wo; α=1e-2)
    for i=1:iters
        x, y = next_batch(loader, batch_size)
        dWx, dWh, dWo = nn_backward(Wx, Wh, Wo, x, y)
        # Vanilla SGD
        Wx += α * dWx
        Wh += α * dWh
        Wo += α * dWo

        if i % 100 == 0
            println("Loop iter $i")
            println("Generated text ...")
            ids = generate_text(200)
            for i in ids
                print(loader.id2char[i])
            end
            println("\n")
        end
    end
end

accuracy(ypred, ytrue) = mean(ypred .== ytrue)
