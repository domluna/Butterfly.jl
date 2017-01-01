abstract AbstractCell{T}

type RNNCell{T<:AbstractFloat} <: AbstractCell{T}
  Wh::AbstractMatrix{T}
  Wx::AbstractMatrix{T}
  Wo::AbstractMatrix{T}
  ht
end

function compute(c::RNNCell, xt)
    xt = reshape(xt, (1, input_size))
    ht = tanh.(xt * c.Wx .+ c.ht * c.Wh)
    return c.ht * c.Wo
end

type LSTMCell{T<:AbstractFloat} <: AbstractCell
end

function compute(c::LSTMCell, x, h)
end
