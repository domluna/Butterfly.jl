type CharLoader
    filepath::AbstractString
    cur_id::Int
    data::AbstractString
    chars::Set{Char}
    id2char::Dict{Int,Char}
    char2id::Dict{Char, Int}
end

function CharLoader(filepath)
    data = readstring(filepath)
    chars = Set(data)
    id2char = Dict(i=>c for (i,c) in enumerate(chars))
    char2id = Dict(c=>i for (i,c) in enumerate(chars))
    return CharLoader(filepath, 1, data, chars, id2char, char2id)
end

function next_batch(loader::CharLoader, batch_size)
    x = zeros(Float32, batch_size, length(loader.chars))
    y = zeros(Int, batch_size)
    local seq::AbstractString
    if loader.cur_id + batch_size > length(loader.data)
        diff = loader.cur_id + batch_size - length(loader.data)
        padding = repeat(" ", diff)
        seq = loader.data[loader.cur_id:length(loader.data)] * padding
        loader.cur_id = 1
    else
        seq = loader.data[loader.cur_id:loader.cur_id+batch_size]
        loader.cur_id += batch_size
    end
    for j=2:length(seq)
        cx = seq[j-1]
        cy = seq[j]
        x[j-1, loader.char2id[cx]] = 1
        y[j-1] = loader.char2id[cy]
    end
    x, y
end
