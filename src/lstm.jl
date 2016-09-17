
# type Gate{T<:AbstractArray}
#     Wx::T
#     Wh::T
#     b::T
# end

# function lstm_cell(input::Gate, forget::Gate, output::Gate, candidate::Gate, ct, ht)
#     ft = sigmoid(xt * forget.Wx + ht * forget.Wh .+ forget.b)
#     it = sigmoid(xt * input.Wx + ht * input.Wh .+ input.b)
#     ot = sigmoid(xt * output.Wx + ht * output.Wh .+ output.b)
#     candidate_memory = tanh(xt * candidate.Wx + ht * candidate.Wh .+ candidate.b)
#     ct = ft * ct + it * candidate_memory
#     ht = ot * tanh(ct)
#     return (ht, ct, ht)
# end