require 'ml-math'


class LinearRegressionModelL2
  def initialize reg_param
    @reg_param = reg_param
  end

  def predict row, w
    x = row["features"]
    yhat = dot(w, x)
  end

  def adjust w
    w.each_key {|k| w[k] = 0.0 if w[k].nan? or w[k].infinite?}
    w.each_key {|k| w[k] = 0.0 if w[k].abs > 1e5 }
  end

  def func examples, w
    # BEGIN YOUR CODE
    loss = 0
    for i in examples
      number = (predict(i, w) - i["label"]).abs()
      number = (number**2) * 0.5
      loss = loss + number
    end
    loss = loss/(examples.size)
    return loss + (norm(w).abs()**2)*(@reg_param/2.0)
    # END YOUR CODE
  end

  def grad examples, w
    g = Hash.new {|h,k| h[k] = 0.0}
    # BEGIN YOUR CODE
    total = Hash.new {|h,k| h[k] = 0.0}
    for i in examples
      for l in i["features"]
        number = (predict(i, w) - i["label"])
        number = number * i["features"][l[0]]
        total[l[0]] = total[l[0]] + number
      end
    end
    for i in w
      weightTotal = weightTotal + w[i[0]]
    end
    for l in total
      g[l[0]] = (total[l[0]]/examples.size) + (@reg_param * weightTotal)
    end
    # END YOUR CODE
    return g
  end
end

class LogisticRegressionModelL2
  def initialize reg_param
    @reg_param = reg_param
  end

  def predict row, w
    x = row["features"]
    1.0 / (1 + Math.exp(-dot(w, x)))
  end

  def adjust w
    w.each_key {|k| w[k] = 0.0 if w[k].nan? or w[k].infinite?}
    w.each_key {|k| w[k] = 0.0 if w[k].abs > 1e5 }
  end

  def func data, w
    # BEGIN YOUR CODE
    loss = 0
    test = 0
    for i in data
      label = -1.0
      if i["label"] > 0
        label = 1.0
      end
      number = 1 + Math.exp(-label*dot(w,i["features"]))
      loss = loss + Math.log(number)
    end
    loss = loss/(data.size)
    return loss + (norm(w).abs()**2)*(@reg_param/2.0)
    # END YOUR CODE
  end

  def grad data, w
    # BEGIN YOUR CODE
    g = Hash.new {|h,k| h[k] = 0.0}
    total = Hash.new {|h,k| h[k] = 0.0}
    for i in data
      label = 0.0
      if i["label"] > 0
        label = 1.0
      end
      for l in i["features"]
        number = (predict(i, w) - label)
        number = number * i["features"][l[0]]
        total[l[0]] = total[l[0]] + number
      end
    end
    for l in total
      g[l[0]] = (total[l[0]]/data.size) + (@reg_param * w[l[0]])
    end
    # END YOUR CODE
    return g
  end
end
# END YOUR CODE


