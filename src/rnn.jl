using ReverseDiffPrototype
const RDP = ReverseDiffPrototype

# input vector is the length of the one-hot encoding or embedding output

# number of unique chars in the dataset
input_size = 97
output_size = 97
state_size = 40
param_scaler = 0.01

Wh = randn(state_size, state_size) * param_scaler
Wx = randn(input_size, state_size) * param_scaler
Wo = randn(state_size, output_size) * param_scaler

# batch_size x state_size,  batch_size x state_size
# 1x97 * 97x40 + 1x40 * 40x40
# 1x40 + 1x40
function rnn_cell(Wx, Wh, Wo, xt, ht)
    xt = reshape(xt, (1, input_size))
    ht = tanh(xt * Wx + ht * Wh) # 1x40
    ot = ht * Wo # 1x97
    return ot, ht # 1x97, 1x40
end

function softmax(logits)
    exp_logits = exp(logits)
    return exp_logits ./ sum(exp_logits, 2)
end

logsoftmax(logits) = log(softmax(logits))

function neural_net(Wx, Wh, Wo, inputs, xt, ht)
    T = size(inputs, 1)
    outputs = zeros(T, output_size)
    ht = zeros(1, state_size)
    for t=1:T
        xt = inputs[t, :]
        ot, ht = rnn_cell(Wx, Wh, Wo, xt, ht)
        outputs[t, :] = ot
    end
    return outputs
end

# TODO: figure out y
function train_neural_net(Wx, Wh, Wo, X, Y)
    loss = 0.
    n = size(X, 1)
    for i=1:n
        outputs = neural_net(Wx, Wh, Wo, X[i, :, :])
        println(typeof(outputs))
        y = Y[i, :]
        inds = zip(1:length(y), y)
        logprobs = logsoftmax(outputs)
        loss += sum(map(i -> -logprobs[i[1], i[2]], inds))
    end
    return loss / n
end

# TODO: preds will be a long vector should it be a matrix?
function predict_neural_net(Wx, Wh, Wo, X)
    preds = Int[]
    n = size(X, 1)
    for i=1:n
        outputs = neural_net(Wx, Wh, Wo, X[i, :, :])
        for i=1:size(outputs,1)
            push!(preds, indmax(outputs[i, :]))
        end
    end
    return preds
end

function nn_backward(Wx, Wh, Wo, X, Y)
    dWx, dWh, dWo, dX = RDP.gradient((Wx, Wh, Wo, X) -> train_neural_net(Wx, Wh, Wo, X, Y), (Wx, Wh, Wo, X))
    return  dWx, dWh, dWo
end

accuracy(ypred, ytrue) = mean(ypred .== ytrue)


# warmup, see if grads make sense
# @time nn_backward(Wx, Wh, Wo, X, Y)
# gc()
# @time dWx, dWh, dWo = nn_backward(Wx, Wh, Wo, X, Y)
# println(dWx[1, 1:5])

# α = 1e-3
# for i in 1:10000
#     x, y = next_batch(loader, 32)
#     dWx, dWh, dWo = nn_backward(Wx, Wh, Wo, X, Y)
#     # Vanilla SGD
#     Wx += α * dWx
#     Wh += α * dWh
#     Wo += α * dWo

#     if i % 1000 == 0
#         ypred = predict_neural_net(W1, testx)
#         acc = accuracy(ypred, testy)
#         println("Loop iter $(i), accuracy = $(acc)")
#     end
# end