require 'ml-interfaces'
require 'transforming-learners'

class LogisticRegressionLearner
  include Learner
  attr_reader :parameters
  attr_accessor :weights

  def initialize regularization: 0.0, learning_rate: 0.01, batch_size: 20, epochs: 1
    @parameters = {"regularization" => regularization,
                   "learning_rate" => learning_rate,
                   "epochs" => epochs, "batch_size" => batch_size}
    print(epochs, " :regulaization\n")
  end

  def train dataset
    @weights = Hash.new {|h,k| h[k] = 0.0}
    @model = LogisticRegressionModelL2.new @parameters["regularization"]
    sgd = StochasticGradientDescent.new @model, @weights, @parameters["learning_rate"]

    # BEGIN YOUR CODE
    for i in 1..@parameters["epochs"]
      examples = dataset["data"].sample(@parameters["batch_size"])
      sgd.update(examples)
    end
    @weights = sgd.weights
    print(@weights, " new weight\n")
    # END YOUR CODE
  end

  def evaluate eval_dataset
    # BEGIN YOUR CODE
    # BEGIN YOUR CODE
    scores = []
    index = 0
    for i in eval_dataset["data"]
      scores << []
      scores[index] << @model.predict(i, @weights)
      scores[index] << i["label"]
      index += 1
    end
    # END YOUR CODE
    return scores
    # END YOUR CODE
  end

  def predict eval_dataset
    #print(eval_dataset, " eval_dataset\n")
    #print(@weights, " @weights\n")
    x = eval_dataset["features"]
    #print(-dot(x, @weights), " negative predict\n")
    #print(Math.exp(-dot(x, @weights)), " exp result\n")
    return @model.predict(eval_dataset, @weights)
  end
end

class SupportVectorMachineLearner
  include Learner
  attr_reader :model
  def initialize complexity: 1.0, kernel: LinearKernel.new
    @parameters = {"complexity" => complexity, "kernel" => kernel}
  end

  def train dataset
    feature_to_id, id_to_feature = create_feature_maps dataset
    libsvm_examples, libsvm_labels = dataset_to_libsvm dataset, feature_to_id

    problem = Libsvm::Problem.new
    parameter = Libsvm::SvmParameter.new

    parameter.cache_size = 1
    parameters["kernel"].update_parameter(parameter)
    parameter.eps = 0.001
    parameter.c = @parameters["complexity"]

    problem.set_examples(libsvm_labels, libsvm_examples)

    filename = "libsvm-#{rand(1e9).to_i.to_s(36)}.model"
    begin
      Libsvm::Model.train(problem, parameter).save(filename)
      @model = load_support_vectors(filename, id_to_feature)
    ensure
      File.delete filename if File.exists? filename
    end
  end

  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] > 0 ? 1 : 0
      [score, label]
    end
  end

  def predict example
    # BEGIN YOUR CODE
    weights = Hash.new {|h,k| h[k] = 0.0}
    weights["bias"] = @model["bias"]
    answer = 0.0
    for i in  @model["data"]
      answer = answer + (i["alpha"]*i["label"]*(@parameters["kernel"].func(example, i)))
    end

    print(answer, " answer\n")
    return answer + @model["bias"]
    # END YOUR CODE
  end
end

class GaussianKernel
  def initialize gamma
    @gamma = gamma
  end

  def update_parameter parameter
    parameter.kernel_type = Libsvm::KernelType::RBF
    parameter.gamma = @gamma
  end

  def func x_i, x_j
    dist = 0
    x = x_i["features"]
    y = x_j["features"]
    for i in x
      if y.has_key?(i[0]) and y[i[0]].is_a?Numeric and  x[i[0]].is_a?Numeric
        dist = dist + (x[i[0]] - y[i[0]]).abs()**2
      else
        dist = dist +(x[i[0]]).abs()**2
      end
    end
    for i in y
      if x.has_key?(i[0]) == false
        dist = dist + (y[i[0]]).abs()**2
      end
    end
    return Math.exp(-@gamma * dist)
  end
end

class NeuralNetworkLearner
  include Learner
  attr_reader :parameters
  attr_accessor :weights

  def initialize batch_size: 20, epochs: 1, problem: nil
    @parameters = {
                   "epochs" => epochs, "batch_size" => batch_size}
    @problem = problem
  end

  def train dataset
    @model = NeuralNetwork.new @problem["layers"]
    sgd = StochasticGradientDescent.new @model, @problem["weights"], @problem["learning_rate"]
    cumulative_loss = 0.0
    i = 0

    # BEGIN YOUR CODE
    for l in 1..@parameters["epochs"]
        examples = dataset["data"].sample(@parameters["batch_size"])
        sgd.update(examples)
        cumulative_loss += obj.func(batch, sgd.weights)
        i += 1
        #puts [i, cumulative_loss / i].join("\t") if i % 100 == 0
      break if cumulative_loss / i < 0.4
    end
    @weights = problem["weights"]
    # END YOUR CODE
  end

  def evaluate eval_dataset
    # BEGIN YOUR CODE
    scores = []
    index = 0
    for i in eval_dataset["data"]
      scores << []
      scores[index] << @model.predict(i)
      scores[index] << i["label"]
      index += 1
    end
    # END YOUR CODE
    return scores
    # END YOUR CODE
  end

  def predict eval_dataset
    x = eval_dataset["features"]
    return @model.predict(eval_dataset)
  end
end

