require 'ml-interfaces'
require 'ml-math'

def cross_validate dataset, folds, &block
  examples = dataset["data"]
  # BEGIN YOUR CODE
  size = dataset["data"].size/folds

  for i in 0..(folds-1)
    train_data = dataset.clone
    test_data = dataset.clone
    train_data["data"] = train_data["data"] - train_data["data"][i * size, size]
    test_data["data"] = test_data['data'][i * size, size]

    print(test_data["data"][0], " first test data for cross\n")
    yield train_data, test_data, i
  end
  # END YOUR CODE
end

class AUCMetric
  # BEGIN YOUR CODE
  include Metric
  def roc_curve(scores)
    fp_rates = [0.0]
    tp_rates = [0.0]
    auc = 0.0

    # BEGIN YOUR CODE
    total_negative = num_negatives(scores)
    total_positive = num_positives(scores)
    cum_negative = 0.0
    cum_positive = 0.0
    new_true_positive = 0.0
    new_false_positive = 0.0
    sortedScores = scores.sort_by {|a,b| a}.reverse!

    for i in sortedScores
      if i[1] == 1
        cum_positive += 1
      else
        cum_negative += 1
      end
      new_false_positive = cum_negative/total_negative
      new_true_positive = cum_positive/total_positive
      auc = auc + (1/2.0)*(new_false_positive - fp_rates[fp_rates.size-1] - 0.0)*(new_true_positive + tp_rates[tp_rates.size-1] - 0.0)
      fp_rates << new_false_positive
      tp_rates << new_true_positive
    end
    # END YOUR CODE
    return [fp_rates, tp_rates, auc]
  end

  def apply scores
    fp, tp, auc = roc_curve scores
    return auc
  end
  # END YOUR CODE
end


# Add any evaluation here. All code must be your own.
# BEGIN YOUR CODE
def num_positives scores
  # BEGIN YOUR CODE
  num = 0.0
  for i in scores
    if i[1] > 0.0
      num += 1
    end
  end
  return num
  # END YOUR CODE
end

def num_negatives scores
  # BEGIN YOUR CODE
  num = 0.0
  for i in scores
    if i[1] <= 0.0
      num += 1
    end
  end
  return num
  # END YOUR CODE
end
# END YOUR CODE


