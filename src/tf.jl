using TensorFlow
const tf = TensorFlow
using Distributions

sess = tf.Session()

x = tf.constant(Float64[1, 2])
y = tf.Variable(Float64[3, 4])
z = tf.placeholder(Float64)

w = exp(x + z + -y)

run(sess, tf.initialize_all_variables())
res = run(sess, w, Dict(z => Float64[1,2]))

# I like the idea of lego blocks
#

"""
A Node can be thought of as a lego block
"""
abstract AbstractNode{I,O}

type Input <: AbstractNode
  x::tf.placeholder
end

Input(dtype; shape=nothing, name="") = Input(tf.placeholder(shape, shape=shape, name=name))

type Dense <: AbstractNode
  inputs
  W::tf.Variable
  b::tf.Variable
end

function Dense(outdim)
  lego = Dense()
end

function compute(n::AbstractNode)
end

type Model
  inputs
  outputs
end
