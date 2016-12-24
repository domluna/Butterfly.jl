using ReverseDiff
using MNIST

type Dataset
  X::AbstractArray
  y::AbstractArray
end

type MNISTData
  train::Dataset
  dev::Dataset
  test::Dataset
end

function onehot{T}(y::Vector{T}, n)
  yy = zeros(T, n, length(y))
  for i=1:length(y)
    label = y[i]
    yy[Int(label)+1, i] = 1
  end
  return yy
end

function load_mnist()
  X, y = MNIST.traindata()
  X_train, y_train = X[:, 1:50000], y[1:50000]
  X_dev, y_dev = X[:, 50001:end], y[50001:end]
  X_test, y_test = MNIST.testdata()
  return MNISTData(
    Dataset(X_train/255, onehot(y_train, 10)),
    Dataset(X_dev/255, onehot(y_dev, 10)),
    Dataset(X_test/255, onehot(y_test, 10))
  )
end

function sgd_update(v, Δv, α)
  for i in eachindex(v)
    v[i] -= α * Δv[i]
  end
end

data = load_mnist()

const BATCH_SIZE = 100
const STEPS_PER_EPOCH = floor(size(data.train.X, 2) / BATCH_SIZE)
const EPOCHS = 20
α = 1e-3

# parameters
W = randn(Float64, 10, 784) * 0.01
b = zeros(Float64, 10)

inputs = (zeros(Float64, 784, BATCH_SIZE), zeros(Float64, 10, BATCH_SIZE), W, b)
outputs = map(similar, inputs)

ReverseDiff.@forward minus_log(x) = -log(x)
softmax(x) = (exp_x = exp.(x); exp_x ./ sum(exp_x, 1))
cross_entropy_loss(x, y) = mean(sum(y .* minus_log.(x), 1))

model(x, y, W, b) = cross_entropy_loss(softmax(W*x .+ b), y)
predict(x, W, b) = W*x .+ b
accuracy(y, ŷ) = mean(mapslices(indmax, y, 1) .== mapslices(indmax, ŷ, 1))

@time model! = ReverseDiff.compile_gradient(model, inputs)

for i=1:EPOCHS
  @time for j=1:STEPS_PER_EPOCH
    s = Int((j-1) * BATCH_SIZE + 1)
    e = Int(j * BATCH_SIZE)
    X_batch = @view data.train.X[:, s:e]
    y_batch = @view data.train.y[:, s:e]
    inputs = (X_batch, y_batch, W, b)
    model!(outputs, inputs)
    sgd_update(W, outputs[3], α)
    sgd_update(b, outputs[4], α)
  end

  dev_acc = accuracy(predict(data.dev.X, W, b), data.dev.y)
  println("Epoch $(i), Dev accuracy = $(dev_acc)")
end

test_acc = accuracy(predict(data.test.X, W, b), data.test.y)
println("Test accuracy = $(test_acc)")
