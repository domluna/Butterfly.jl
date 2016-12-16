using ReverseDiff
using CatViews

x = [-1. -2; -1 -2]
y = [1.; 2]
w = [2.; 3]
b = [-3.]

type Vars
    lookup::Dict
    grad_lookup::Dict
    flat
    grad_flat

    Vars() = Vars(Dict(), Dict(), nothing, nothing)
end
getindex(vars::Vars, varname::String) = vars.lookup[varname]
setindex!(vars::Vars, varname::String, var) = vars.lookup[varname] = var
function compile!(vars::Vars)
    for (k, v) in vars.lookup
        vars.grad_lookup[k] = similar(v)
    end
    vars.flat = CatView(map(x -> view(x, :), values(vars.lookup)))
    vars.grad_flat = CatView(map(x -> view(x, :), values(vars.grad_lookup)))
end


vars = Vars()
vars[:w] = w
vars[:b] = b
compile!(vars)

ReverseDiff.@forward sigmoid(x) = 1. ./ (1. + exp(-x))
mse(a, y) = mean((a - y) .* (a - y))
#= mse(a, y) = mean((a - y) .^ 2) =#
f1(x, y, vars::Vars) = mse(sigmoid(x*vars[:w] .+ vars[:b]), y)

out = f1(x, y, vars)
println(out)

inputs = (x, y, vars)
result = map(similar, inputs)
result_view = CatView(map(x -> view(x, :), result))
f1! = ReverseDiff.compile_gradient(f1, result)

@time f1!(result, inputs)
@time f1!(result, inputs)

for r in result
    println(r)
end

println(result_view)

#= Expected output =#
#=  =#
#= [array([[ -3.34017280e-05,  -5.01025919e-05], =#
#=        [ -6.68040138e-05,  -1.00206021e-04]]), array([[ 0.9999833], =#
#=        [ 1.9999833]]), array([[  5.01028709e-05], =#
#=        [  1.00205742e-04]]), array([ -5.01028709e-05])] =#
