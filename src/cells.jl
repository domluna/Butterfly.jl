abstract AbstractCell

type RNNCell{T} <: AbstractCell
    
end

function compute(::RNNCell, x, h)
end

type LSTMCell{T} <: AbstractCell
end

function compute(::LSTMCell, x, h)
end

type GRUCell{T} <: AbstractCell
end

function compute(::GRUCell, x, h)
end