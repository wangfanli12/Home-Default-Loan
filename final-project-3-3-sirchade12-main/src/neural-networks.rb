require "distribution"
include Distribution

def connect src, dst
  dst.back_net[src.name] = src
end


class Input
  attr_accessor :activation
  attr_reader :back_net
  attr_reader :name

  def initialize name
    @name = name
    @back_net = nil
  end

  def forward
    # BEGIN YOUR CODE
    return @activation
    # END YOUR CODE
  end

  def pderv name
    # BEGIN YOUR CODE
    if name == @name
      return 1.0
    else
      return 0.0
    end
    # END YOUR CODE
  end
end

class L2Loss
  attr_reader :activation
  attr_reader :back_net
  attr_reader :name
  attr_accessor :feedback

  def initialize
    @back_net = Hash.new
  end

  def incoming
    @back_net.values.first
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].activation
    end
    @activation = total
    return @activation
    # END YOUR CODE
  end

  def loss
    # BEGIN YOUR CODE
    total = 0.0
    total = total + (((@activation - feedback).abs()**2)*0.5)
    return total/(@back_net.size + 0.0)
    # END YOUR CODE
  end

  def pderv fname
    # BEGIN YOUR CODE
    total = 0.0
    size = 0.0
    for i in @back_net
      if fname == i[1].name
        total = total + (i[1].activation - feedback)
        size += 1.0
      end
    end
    if (size == 0)
      total = 0.0
      for i in @back_net
        total = total + i[1].pderv(fname) * (i[1].activation - feedback)
      end
      return total
    end
    return total/(size + 0.0)
    # END YOUR CODE
  end
end

class LogLoss
  attr_reader :activation
  attr_reader :back_net
  attr_reader :name
  attr_accessor :feedback

  def initialize
    @back_net = Hash.new
  end

  def incoming
    @back_net.values.first
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].activation
    end
    @activation = 1/ (1.0 + Math.exp(-total))
    return @activation
    # END YOUR CODE
  end

  def pderv fname
    # BEGIN YOUR CODE
    total = 0.0
    size = 0.0
    feedback1 = feedback
    if feedback1 == -1
      feedback1 = 0
    end
    for i in @back_net
      if fname == i[1].name
        total = total + ((1/ (1.0 + Math.exp(-i[1].activation))) - feedback1)
        size += 1.0
      end
    end
    if (size == 0)
      total = 0.0
      for i in @back_net
        total = total + i[1].pderv(fname) * ((1/ (1.0 + Math.exp(-i[1].activation))) - feedback1)
      end
      return total
    end
    return total/(size + 0.0)
    # END YOUR CODE
  end

  def loss
    # BEGIN YOUR CODE
    total = 0.0
    feedback1 = feedback
    if feedback1 == -1
      feedback1 = 0.0
    end
    total = total + ((feedback1 * Math.log(@activation)) +
        (1 - feedback1)*(Math.log(1.0 - @activation)))
    return -total
    # END YOUR CODE
  end
end

class Sigmoid
  attr_reader :activation
  attr_reader :name
  attr_reader :back_net

  def initialize name = "_"
    @name = name
    @back_net = Hash.new
  end

  def incoming
    @back_net.values.first
  end

  def sig x
    # BEGIN YOUR CODE
    return 1/ (1.0 + Math.exp(-x))
    # END YOUR CODE
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].activation
    end
    @activation = sig(total)
    return @activation
    # END YOUR CODE
  end

  def pderv fname
    # BEGIN YOUR CODE
    total = 0.0
    size = 0.0
    for i in @back_net
      if fname == i[1].name
        total = total + (sig(i[1].activation)) * ( 1 - sig(i[1].activation))
        size += 1.0
      end
    end
    if (size == 0)
      total = 0.0
      for i in @back_net
        total = total + i[1].pderv(fname) * (sig(i[1].activation)) * ( 1 - sig(i[1].activation))
      end
      return total
    end
    return total
    # END YOUR CODE
  end
end

class Tanh
  attr_reader :activation
  attr_reader :name
  attr_reader :back_net

  def initialize name = "_"
    @name = name
    @back_net = Hash.new
  end

  def incoming
    @back_net.values.first
  end

  def tanh x
    # BEGIN YOUR CODE
    return ((Math.exp(x) - Math.exp(-x))/ (Math.exp(x) + Math.exp(-x)))
    # END YOUR CODE
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].activation
    end
    @activation = tanh(total)
    return @activation
    # END YOUR CODE
  end

  def pderv fname
    # BEGIN YOUR CODE
    total = 0.0
    size = 0.0
    for i in @back_net
      if fname == i[1].name
        total = total + (1 - ((tanh(i[1].activation)).abs()**2))
        size += 1.0
      end
    end
    if (size == 0)
      total = 0.0
      for i in @back_net
        total = total + i[1].pderv(fname) * (1 - ((tanh(i[1].activation)).abs()**2))
      end
    end
    return total
    # END YOUR CODE
  end
end

class ReLU
  attr_reader :activation
  attr_reader :name
  attr_reader :back_net

  def initialize name
    @name = name
    @back_net = Hash.new
  end

  def incoming
    @back_net.values.first
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].activation + 0.0
    end
    #print(@back_net, " back_net during max\n")
    @activation = [0.01*total, total].max()
    return @activation
    # END YOUR CODE
  end

  def pderv fname
    # BEGIN YOUR CODE
    total = 0.0
    size = 0.0
    #print(@back_net, " back net for back proga\n")
    for i in @back_net
      if fname == i[1].name
        value = 0
        if i[1].activation <= 0
          value = 0.01
        else
          value = 1.0
        end
        total = total + value
        size += 1.0
      end
    end
    if (size == 0)
      total = 0.0
      for i in @back_net
        value = 0
        if i[1].activation <= 0
          value = 0.01
        else
          value = 1.0
        end
        #print(i[1].name, " name of the cell\n")
        if i[1].pderv(fname).nil? == true
          print(@back_net, " the whole back_net\n")
        end
        total = total + i[1].pderv(fname) * value
      end
      return total
    end
    return total/(size + 0.0)
    # END YOUR CODE
  end
end

class LinearUnit
  attr_reader :activation
  attr_reader :back_net
  attr_reader :name
  attr_reader :weights

  def initialize name, weights
    @name = name
    @weights = weights
    @back_net = Hash.new
  end

  def n fname
    [@name, fname].join(".")
  end

  def forward
    # BEGIN YOUR CODE
    total = 0.0
    #print(@name, "name of unit\n")
    #print(@back_net, " back_net\n")
    for i in @back_net
      #print(@weights, " neural weights\n")
      total += i[1].activation * @weights[n(i[1].name)]
    end
    total += @weights[n("bias")]
    @activation = total
    # END YOUR CODE
    return @activation
  end

  def pderv_weights name
    # BEGIN YOUR CODE
    if name == n("bias")
      return 1.0
    end
    for i in @back_net
      if n(i[1].name) == name
        return i[1].activation
      end
    end
    return 0.0
    # END YOUR CODE
  end

  def pderv_back name
    # BEGIN YOUR CODE
    total = 0.0
    for i in @back_net
      total = total + i[1].pderv(name) * @weights[n(i[1].name)]
    end
    return total
    # END YOUR CODE
  end


  def pderv name
    if name.start_with?(@name + ".") and @weights.has_key? name
      pderv_weights name
    else
      pderv_back name
    end
  end
end

class LogisticRegression
  def initialize weights
    @x1 = Input.new "x1"
    @x2 = Input.new "x2"
    @inner = LinearUnit.new "wx", weights
    @sig = Sigmoid.new "sig"
    @out = L2Loss.new

    connect @x1, @inner
    connect @x2, @inner
    connect @inner, @sig
    connect @sig, @out
  end

  def predict example
    # BEGIN YOUR CODE
    @x1.activation = example["features"][@x1.name]
    @x2.activation = example["features"][@x2.name]
    @x1.forward
    @x2.forward
    @inner.forward
    @sig.forward
    #print(result, " predict result\n")
    return @out.forward
    # END YOUR CODE
  end

  def predict_batch examples
    examples.collect {|row| forward row}
  end

  def forward row
    # BEGIN YOUR CODE
    @x1.activation = row["features"][@x1.name]
    @x2.activation = row["features"][@x2.name]
    @out.feedback = row["label"]
    @x1.forward
    @x2.forward
    @inner.forward
    @sig.forward
    @out.forward
    # END YOUR CODE
  end

  def func dataset, weights
    # BEGIN YOUR CODE
    @inner = LinearUnit.new "wx", weights
    connect @x1, @inner
    connect @x2, @inner
    connect @inner, @sig

    loss = 0.0
    for i in dataset
      forward(i)
      loss += @out.loss()
    end
    loss = loss/((dataset.size) + 0.0)
    return loss
    # END YOUR CODE
  end

  def grad data, weights
    g = Hash.new {|h,k| h[k] = 0.0}
    # BEGIN YOUR CODE
    @inner = LinearUnit.new "wx", weights
    connect @x1, @inner
    connect @x2, @inner
    connect @inner, @sig

    total = Hash.new {|h,k| h[k] = 0.0}
    #print(weights, " weights\n")
    for i in data
      forward(i)
      label = 0.0
      if i["label"] > 0
        label = 1.0
      end
      for l in i["features"]
        #number = (predict(i) - label)
        #number = number * i["features"][l[0]]
        #total[l[0]] = total[l[0]] + number
        temp = l[0]
        #print(@inner.pderv(@inner.n(l[0])), " inner pderv\n")
        #print(@sig.pderv(l[0]), " sig pderv\n")
        #print(@out.pderv(l[0]), " out pderv\n")
        total[@inner.n(l[0])] = total[@inner.n(l[0])] + (@inner.pderv(@inner.n(l[0])) * @sig.pderv(@inner.name) * @out.pderv(@sig.name))
      end
      total[@inner.n("bias")] = total[@inner.n("bias")] + (@inner.pderv(@inner.n("bias")) * @sig.pderv(@inner.name) * @out.pderv(@sig.name))
    end
    for l in total
      g[l[0]] = total[l[0]]
    end
    #print(g, " the gradient\n")
    return g
    # END YOUR CODE
  end

  def adjust weights
  end
end


class NeuralNetwork
  attr_reader :batch_loss
  def initialize layers
    @inputs = Hash.new
    @layers = layers
    @out = @layers[-1][0]
    @batch_loss = 0.0
    #print(@layers, " layers\n")
  end

  def forward example
    # BEGIN YOUR CODE
    for i in @inputs
      i[1].activation = example["features"][i[1].name]
    end
    @out.feedback = example["label"]
    for i in @layers
      for l in i
        l.forward()
      end
    end
    # END YOUR CODE
  end

  def func examples, weights
    # BEGIN YOUR CODE
    loss = 0.0
    for i in examples
      forward(i)
      loss += @out.loss()
    end
    loss = loss/((examples.size) + 0.0)
    return loss
    # END YOUR CODE
  end

  def grad examples, weights
    update_inputs examples
    # BEGIN YOUR CODE
    total = Hash.new {|h,k| h[k] = 0.0}
    for i in examples
      forward(i)
      for act in @layers
        for node in act
          if node.instance_of?(LinearUnit)
            for edge in node.back_net
              total[node.n(edge[0])] = total[node.n(edge[0])] + (@out.pderv(node.n(edge[0])))
            end
            total[node.n("bias")] = total[node.n("bias")] + (@out.pderv(node.n("bias")))
          end
        end
      end
    end
    return total
    # END YOUR CODE
  end

  def predict_batch examples
    examples.collect {|example| predict example}
  end

  def predict example
    # BEGIN YOUR CODE
    for i in @inputs
      i[1].activation = example["features"][i[1].name]
    end
    for i in @layers
      for l in i
        l.forward()
      end
    end
    return @out.forward()
    # END YOUR CODE
  end

  def update_inputs examples
    examples.flat_map {|r| r["features"].keys}
    .uniq
    .reject {|k| @inputs.has_key? k}
    .each do |k|
      @inputs[k] = Input.new k
      @layers.first.each {|f| connect @inputs[k], f}
    end
    @inputs
  end

  def adjust weights
  end
end

def create_problem learning_rate, seed, layout
  # BEGIN YOUR CODE
  alpha = "abcdefghijklmnopq"
  rng = Normal.rng(0,1, seed)
  problem_xor = Hash.new
  weights = Hash.new {|h,k| h[k] = rng.call()}
  problem_xor["weights"] = weights
=begin
  layers = [
      [LinearUnit.new("w1", weights), LinearUnit.new("w2", weights), LinearUnit.new("w3", weights), LinearUnit.new("w4", weights)],
      [ReLU.new("sig1"), ReLU.new("sig2"), ReLU.new("sig3"), ReLU.new("sig4")],
      [LinearUnit.new("final", weights)],
      [LogLoss.new]
  ]
=end
  layers = Array.new
  index = 0.0
  alphaIndex = 0.0
  for i in layout
    layers[index] = Array.new
    for l in 0..i-1
      layers[index].append(LinearUnit.new(alpha[alphaIndex] + l.to_s(), weights))
      weights[layers[index][l].n("bias")] = 0.0
    end
    index += 1
    layers[index] = Array.new
    for l in 0..i-1
      layers[index].append(Tanh.new("Tanh" + alpha[alphaIndex] + l.to_s()))
    end
    alphaIndex += 1
    index += 1
  end

  print(layers, " the constructed layers before connection\n")
  print(layers.size, " the size of layers\n")

  for i in 1..layers.size-1
    if layers[i][0].is_a?(Tanh)
      for l in 0..layers[i].size-1
        connect layers[i-1][l], layers[i][l]
      end
    else
      for l in 0..layers[i].size-1
        for m in 0..layers[i-1].size-1
          connect layers[i-1][m], layers[i][l]
        end
      end
    end
  end
  layers.append([LinearUnit.new("final", weights)])
  weights[layers[-1][0].n("bias")] = 0.0
  layers.append([LogLoss.new])


  for i in 0..layers[-3].size-1
    connect layers[-3][i], layers[-2][0]
  end

  connect layers[-2][0], layers[-1][0]

  print(layers, " the constructed layers\n")

  problem_xor["layers"] = layers
  problem_xor["learning_rate"] = learning_rate
  # END YOUR CODE
  return problem_xor
end

def create_manual_problem learning_rate, seed, layout
  # BEGIN YOUR CODE
  alpha = "abcdefghijklmnopq"
  rng = Normal.rng(0,1, seed)
  problem_xor = Hash.new
  weights = Hash.new {|h,k| h[k] = rng.call()}
  problem_xor["weights"] = weights

  layers = [
      [LinearUnit.new("w1", weights), LinearUnit.new("w2", weights), LinearUnit.new("w3", weights), LinearUnit.new("w4", weights)],
      [ReLU.new("sigw1"), ReLU.new("sigw2"), ReLU.new("sigw3"), ReLU.new("sigw4")],
      [LinearUnit.new("y1", weights), LinearUnit.new("y2", weights)],
      [ReLU.new("sigy1"), ReLU.new("sigy2")],
      [LinearUnit.new("final", weights)],
      [LogLoss.new]
  ]

  connect layers[0][0], layers[1][0]
  connect layers[0][1], layers[1][1]
  connect layers[0][2], layers[1][2]
  connect layers[0][3], layers[1][3]

  connect layers[1][0], layers[2][0]
  connect layers[1][1], layers[2][0]
  connect layers[1][2], layers[2][0]
  connect layers[1][3], layers[2][0]

  connect layers[1][0], layers[2][1]
  connect layers[1][1], layers[2][1]
  connect layers[1][2], layers[2][1]
  connect layers[1][3], layers[2][1]


  connect layers[2][0], layers[3][0]
  connect layers[2][1], layers[3][1]

  connect layers[3][0], layers[4][0]
  connect layers[3][1], layers[4][0]

  connect layers[4][0], layers[5][0]


  problem_xor["layers"] = layers
  problem_xor["learning_rate"] = learning_rate
  print(layers, " the constructed layers\n")
  # END YOUR CODE
  return problem_xor
end