using ReverseDiffPrototype
const RDP = ReverseDiffPrototype
using Distributions
include("char_loader.jl")

loader = CharLoader("../data/input.txt")

# number of unique chars in the dataset
batch_size = 25
input_size = length(loader.chars)
output_size = length(loader.chars)
state_size = 100
param_scaler = 0.01

Wh = randn(state_size, state_size) * param_scaler
Wx = randn(input_size, state_size) * param_scaler
Wo = randn(state_size, output_size) * param_scaler

function softmax(logits)
    exp_logits = exp(logits)
    return exp_logits ./ sum(exp_logits, 2)
end

logsoftmax(logits) = log(softmax(logits))

function rnn_cell(Wx, Wh, Wo, xt, ht)
    xt = reshape(xt, (1, input_size))
    ht = tanh(xt * Wx + ht * Wh)
    ot = ht * Wo
    return ot, ht
end
 
function train_neural_net(Wx, Wh, Wo, X, Y)
    loss = 0.
    T = size(X, 1)
    ht = zeros(1, state_size)
    for t=1:T
        ot, ht = rnn_cell(Wx, Wh, Wo, X[t, :], ht)
        y = Y[t]
        loss += -(logsoftmax(ot)[y])
    end
    return loss / T
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

function nn_backward(Wx, Wh, Wo, X, Y)
    dWx, dWh, dWo, dX = RDP.gradient((Wx, Wh, Wo, X) -> train_neural_net(Wx, Wh, Wo, X, Y), (Wx, Wh, Wo, X))
    return  dWx, dWh, dWo
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


# x, y = next_batch(loader, batch_size)
# # warmup, see if grads make sense
# @time nn_backward(Wx, Wh, Wo, x, y)
# gc()
# @time dWx, dWh, dWo = nn_backward(Wx, Wh, Wo, x, y)
# println(dWx[1, 1:5])
# println(dWh[1, 1:5])
# println(dWo[1, 1:5])

# train_model(100000, Wx, Wh, Wo)