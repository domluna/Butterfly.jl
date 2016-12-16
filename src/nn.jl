using ReverseDiff
using CatViews

#= x = [-1. -2; -1 -2] =#
#= y = [1.; 2] =#
#= w = [2.; 3] =#
#= b = [-3.] =#

x = randn(32, 784)
y = randn(32, 1)
w1 = randn(784, 100)
b1 = zeros(1, 100)
w2 = randn(100, 1)
b2 = zeros(1)

ReverseDiff.@forward sigmoid(x) = 1. ./ (1. + exp(-x))
mse(a, y) = mean((a - y) .^ 2)

function nn(x, y, w1, b1, w2, b2)
    out1 = sigmoid(x*w1 .+ b1)
    out2 = out1*w2 .+ b2
    return mse(out2, y)
end

function nn(x, y, w, b)
    return mse(sigmoid(x*w .+ b), y)
end

inputs = (x, y, w1, b1, w2, b2)
#= inputs = (x, y, w, b) =#

result = map(similar, inputs)
result_view = CatView(map(x -> view(x, :), result))
@time nn! = ReverseDiff.compile_gradient(nn, result)

@time nn!(result, inputs)
@time nn!(result, inputs)

#= for r in result =#
#=     println(r) =#
#= end =#
#= println(result_view) =#

#= Expected output =#
#=  =#
#= [array([[ -3.34017280e-05,  -5.01025919e-05], =#
#=        [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833], =#
#=        [ 1.9999833]]), array([[  5.01028709e-05], =#
#=        [  1.00205742e-04]]), array([ -5.01028709e-05])] =#
