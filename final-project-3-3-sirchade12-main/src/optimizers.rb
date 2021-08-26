class StochasticGradientDescent
  attr_reader :weights
  attr_reader :objective
  def initialize obj, w_0, lr = 0.01
    @objective = obj
    @weights = w_0
    @n = 1.0
    @lr = lr
  end
  def update x
    # BEGIN YOUR CODE
    grad = @objective.grad(x, @weights)
    newLearn = @lr/Math.sqrt(@n)
    if @objective.is_a?(NeuralNetwork)
      for i in grad
        #print((grad[i[0]]), " grade value\n")
        @weights[i[0]] = @weights[i[0]] - (((grad[i[0]]) / (x.size()+0.0)) * newLearn)
      end
    else
      for i in grad
        @weights[i[0]] = @weights[i[0]] - (grad[i[0]] * newLearn)
      end
    end
    @objective.adjust(@weights)
    @n = @n + 1
    # END YOUR CODE
  end
end
