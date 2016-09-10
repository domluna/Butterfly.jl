using MNIST

type DataLoader
    cur_id::Int
    order::Vector{Int}
end

DataLoader() = DataLoader(1, shuffle(1:60000))

function next_batch(loader::DataLoader, batch_size)
    x = zeros(Float32, batch_size, 784)
    y = zeros(Int, batch_size)
    for i in 1:batch_size
        x[i, :] = trainfeatures(loader.order[loader.cur_id])
        label = trainlabel(loader.order[loader.cur_id])
        y[i] = Int(label)+1
        loader.cur_id += 1
        if loader.cur_id > 60000
            loader.cur_id = 1
        end
    end
    x = (x .- mean(x, 2)) ./ std(x, 2)
    x, y
end

function load_test_set(N=10000)
    x = zeros(Float32, N, 784)
    y = zeros(Float32, N)
    for i in 1:N
        x[i, :] = testfeatures(i)
        label = testlabel(i)
        y[i] = Int(label)+1
    end
    x = (x .- mean(x, 2)) ./ std(x, 2)
    x, y
end
