require 'digest'
require 'ml-math'
require 'ml-interfaces'

# Add any transformers you want here. All code must be your own.


class ZScoreTransformer
  include FeatureTransformer
  attr_reader :means, :stdevs, :feature_names
  def initialize feature_names
    @means = Hash.new
    @stdevs = Hash.new
    @feature_names = feature_names
  end

  def train dataset
    # BEGIN YOUR CODE
    for feature in @feature_names
      goalList = []
      for data in dataset["data"]
        for i in data["features"]
          if i[0] == feature and i[1].is_a?(Numeric) and i[1] == i[1]
            goalList.append(i[1])
          end
        end
      end
      @means[feature] = mean(goalList)
      @stdevs[feature] = stdev(goalList)
    end
    # END YOUR CODE
  end

  def apply example_batch
    # BEGIN YOUR CODE
    example_batch1 = example_batch.clone()
    example_batch1["data"] = example_batch["data"].clone()
    for feature in @feature_names
      for l in example_batch1
        for i in l["features"]
          if feature == i[0] and i[1].nil? == false and i[1].is_a?(Numeric)
            if @stdevs[feature] != 0
              example_batch1["features"][feature] = (example_batch1["features"][feature] - @means[feature])/(@stdevs[feature])
            end
          end
        end
      end
    end
    # END YOUR CODE
  end
end

class MeanImputation
  include FeatureTransformer
  attr_reader :means

  def initialize feature_names
    @means = Hash.new
    @feature_names = feature_names
  end

  def train dataset
    # BEGIN YOUR CODE
    for feature in @feature_names
      goalList = []
      for data in dataset["data"]
        for i in data["features"]
          if i[0] == feature and i[1].is_a?(Numeric) and i[1] == i[1]
            goalList.append(i[1])
          end
        end
      end
      @means[feature] = mean(goalList)
    end
    # END YOUR CODE
  end

  def apply(example_batch)
    # BEGIN YOUR CODE
    for feature in @feature_names
      for l in example_batch
        if l["features"].key?(feature) == false
          l["features"][feature] = @means[feature]
        end
        for i in l["features"]
          if feature == i[0] and (i[1] == nil or i[1] != i[1])
            l["features"][feature] = @means[feature]
          end
        end
      end
    end
    # END YOUR CODE
    return example_batch
  end
end

class AgeRangeAsVector
  include FeatureTransformer
  def initialize; end
  def train dataset; end
  def apply(example_batch)
    min_age = 0
    max_age = 100
    feature_name = "days_birth"
    pattern = "age_range_%d"
    # BEGIN YOUR CODE
    example_clone = example_batch.clone()
    feature = feature_name
    for l in example_batch
      old = 0
      for i in l["features"]
        if feature == i[0] and i[1] != nil and i[1].is_a?(Numeric)
          l["features"][feature] = 5 * (-l["features"][feature]/(365*5))
          if l["features"][feature] > max_age
            l["features"][feature] = max_age
          elsif l["features"][feature] < min_age
            l["features"][feature] = min_age
          end
        end
      end
      old = l["features"].delete(feature)
      if old.nil? == false
        if l["features"].key?(pattern%[old]) == false
          l["features"][pattern%[old]] = 0
        end
        l["features"][pattern%[old]] = l["features"][pattern%[old]] + 1
      end
    end
    # END YOUR CODE
    return example_batch
  end
end

class TargetAveraging
  include FeatureTransformer
  attr_reader :means

  def initialize feature_names
    @means = Hash.new {|h,k| h[k] = Hash.new}
    @feature_names = feature_names
    @pattern = "avg_%s"
  end

  def train dataset
    # BEGIN YOUR CODE
    for feature in @feature_names
      goalList = Hash.new {|h,k| h[k] = Hash.new {|h, k| h[k] = 0}}
      probA = Hash.new {|h,k| h[k] = 0.0}
      for data in dataset["data"]
        for i in data["features"]
          if i[0] == feature and i[1] == i[1] and i[1] != nil
            goalList[i[0]][i[1]] += 1.0
            if data["label"] == 1
              probA[i[1]] += 1.0
            end
          end
        end
      end
      for i in goalList[feature]
        @means[feature][i[0]] = probA[i[0]]/i[1]
      end
    end
    # END YOUR CODE
  end

  def apply(example_batch)
    print("target aveaging now\n")
    # BEGIN YOUR CODE
    example = example_batch.clone()
    for feature in @feature_names
      for l in example_batch
        if l["features"][feature] != nil
          old = l["features"].delete(feature)
          l["features"][@pattern%[feature]] = @means[feature][old]
        end
      end
    end
    # END YOUR CODE
    return example_batch
  end
end


class OneHotEncoding
  include FeatureTransformer
  def initialize feature_names
    @feature_names = feature_names
    @pattern = "%s=%s"
  end

  def train dataset; end

  def apply(example_batch)
    # BEGIN YOUR CODE
    for feature in @feature_names
      for l in example_batch
        if l["features"].key?(feature) == true and l["features"][feature] != nil
          old = l["features"].delete(feature)
          l["features"][@pattern%[feature, old]] = 1.0
        end
      end
    end
    # END YOUR CODE
    return example_batch
  end
end

class LogTransform
  include FeatureTransformer
  def initialize feature_names
    @feature_names = feature_names
    @pattern = "log_%s"
  end

  def train dataset; end

  def apply(example_batch)
    # BEGIN YOUR CODE
    print("log transform now\n")
    for feature in @feature_names
      for l in example_batch
        if l["features"].key?(feature) == true and l["features"][feature].nil? == false and l["features"][feature].is_a?(Numeric)
          old = l["features"].delete(feature)
          l["features"][@pattern%[feature]] = Math.log(old)
        end
      end
    end
    # END YOUR CODE
    return example_batch
  end
end

class L2Normalize
  include FeatureTransformer
  def train dataset; end
  def apply(example_batch)
    # BEGIN YOUR CODE
    for l in example_batch
      normal = norm(l["features"])
      for i in l["features"]
        if i[1] != nil and i[1].is_a?(Numeric)
          l["features"][i[0]] = l["features"][i[0]] / (normal+0.0)
        end
      end
    end
    # END YOUR CODE
    return example_batch
  end
end

class DownsampleNegatives
  include FeatureTransformer
  attr_reader :sampling_rate
  def initialize sampling_rate
    @sampling_rate = sampling_rate
  end

  def train dataset; end

  def update_sampling_rate dataset
    # BEGIN YOUR CODE
    probA = 0
    for data in dataset["data"]
      if data["label"] <= 0
        probA += 1.0
      end
    end
    rate = (dataset["data"].size + 0.0 - probA)/probA
    @sampling_rate = rate
    # END YOUR CODE
  end

  def hashprob id
    salt = "eifjcchdivlbreckvgndlvkgdtdjnbcnjldelrgefcgt"
    (Digest::MD5.hexdigest(id.to_s + salt).to_i(16) % 100000).abs / 100000.0
  end

  def can_keep? example
    can_keep = true
    # BEGIN YOUR CODE
    prob = hashprob(example["id"])
    if prob >= @sampling_rate and example["label"] <= 0
      can_keep =  false
    end

    # END YOUR CODE
    return can_keep
  end

  def apply(example_batch)
    return example_batch.select! {|example| can_keep? example}
  end
end

class FeatureTransformPipeline
  def initialize *transformers
    @transformers = transformers
    raise ArgumentError.new("All transformers must 'include FeatureTransformer' in the class") unless @transformers.all? {|t| t.class.include? FeatureTransformer}
  end

  def train dataset
    # BEGIN YOUR CODE
    for i in @transformers
      i.train(dataset)
      i.apply(dataset["data"])
    end
    # END YOUR CODE
  end

  def apply example_batch
    return @transformers.inject(example_batch) do |u, transform|
      u = transform.apply example_batch
    end

  end
end

    
