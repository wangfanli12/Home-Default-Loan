require 'ml-interfaces'
require 'json'

def class_distribution dataset
  distribution = Hash.new {|h,k| h[k] = 0}
  # BEGIN YOUR CODE
  for i in dataset
    distribution[i["label"]] += 1.0
  end
  for i in distribution
    distribution[i[0]] = distribution[i[0]]/(dataset.size() + 0.0)
  end
  # END YOUR CODE
  return distribution
end

def entropy dist
  # BEGIN YOUR CODE
  #print(dist, " dist\n")
  total = 0.0
  for i in dist
    if (i[1] > 0) and ( i[1] < 1)
      total = total + i[1]* Math.log(i[1])
    else
      total = total + 0.0
    end
  end
  return -total
  # END YOUR CODE
end

def information_gain h0, splits
  # BEGIN YOUR CODE
  #print(h0, " h0\n")
  count = 0.0
  total_entropy = Hash.new {|h,k| h[k] = 0}
  for i in splits
    count = count + i[1].size()
    dist = class_distribution(i[1])
    total_entropy[i[0]] = entropy(dist)
  end
  total = 0.0
  for i in total_entropy
    total = total + (i[1] * (splits[i[0]].size()/(count+0.0)))
  end
  #print(total_entropy, " total entropy\n")
  #print(count, " split size\n")
  #print(total, " total\n")
  return h0 - total
  # END YOUR CODE
end

class NumericSplit
  attr_reader :feature_name, :split_point, :paths
  def initialize fname, value
    @feature_name = fname
    @split_point = value
    @split_point_str = "%.2g" % @split_point
    @paths = ["#{@feature_name} < #{@split_point_str}", "#{@feature_name} >= #{@split_point_str}"]
  end

  def to_s
    "Numeric[#{@feature_name} <=> #{@split_point_str}]"
  end

  def split_on_feature examples
    # BEGIN YOUR CODE

    @path = ["#{@feature_name} < #{@split_point_str}", "#{@feature_name} >= #{@split_point_str}"]
    splits = Hash.new {|h,k| h[k] = []}
    splits[@path[0]] = []
    splits[@path[1]] = []
    feature = 0.0
    for l in examples
      trigger = false
      for i in l["features"]
        if i[0] == @feature_name
          trigger = true
          feature += 1.0
          if i[1] < @split_point
            splits[@path[0]].append(l)
          else
            splits[@path[1]].append(l)
          end
        end
      end
      if trigger == false
        if 0 < @split_point
          splits[@path[0]].append(l)
        else
          splits[@path[1]].append(l)
        end
      end
    end
    # END YOUR CODE
    return splits
  end

  def test example
    # BEGIN YOUR CODE
    for i in example["features"]
      if i[0] == @feature_name
        if i[1] < @split_point
          return "%s < %s" % [i[0], @split_point_str]
        else
          return "%s >= %s" % [i[0], @split_point_str]
        end
      end
    end
    return nil
    # END YOUR CODE

    return path_name
  end
end

class NumericSplitter
  def matches? examples, feature_name
    has_feature = examples.select {|r| r["features"].has_key? feature_name}
    return false if has_feature.empty?
    return has_feature.all? do |r|
      r["features"].fetch(feature_name, 0.0).is_a?(Numeric)
    end
  end

  def create_split examples, parent_entropy, feature_name
    # BEGIN YOUR CODE
    #print(feature_name, " feature_name\n")
    #print(matches?(examples, feature_name), " \n")
    information = []
    if(matches?(examples, feature_name)) == false
      return nil
    end
    clone = examples.clone
    for l in clone
      if l["features"].key?(feature_name) == false
        l["features"][feature_name] = 0.0
      end
    end
    sorted = clone.sort {|a,b| a["features"][feature_name] <=> b["features"][feature_name]}
    dist = Hash.new
    dist["left"] = []
    dist["right"] = []
    for i in sorted
      dist["right"] << i
    end
    lastCutOff = -999999999.0
    for i in sorted
      cutoff = i["features"][feature_name]
      if lastCutOff != cutoff
        information << [information_gain(parent_entropy, dist), cutoff]
      end
      lastCutOff = cutoff
      dist["left"] << i
      dist["right"].shift()
      #print(dist, " dist after one shift\n")
      #print(information, " information after one shift\n")
    end

    information = information.sort{|a,b| a[0] <=> b[0]}.reverse!
    #print(information, " information after sort\n")
    split_point = information[0][1]
    ig = information[0][0]
    #print(split_point, " split point\n")
    # END YOUR CODE

    split = NumericSplit.new(feature_name, split_point)

    return {"split" => split, "information_gain" => ig}
  end
end

class CategoricalSplit
  attr_reader :feature_name

  def initialize fname
    @feature_name = fname
    @path_pattern = "%s == '%s'"
  end

  def to_s
    "Categorical[#{@feature_name}]"
  end

  def split_on_feature examples
    splits = Hash.new

    # BEGIN YOUR CODE
    splits = Hash.new {|h,k| h[k] = []}
    total = 0
    feature = 0
    #@path_pattern % [@feature_name, feature_value]
    for l in examples
      total = total + 1.0
      trigger = false
      for i in l["features"]
        if i[0] == @feature_name and i[1].nil? == false
          trigger = true
          feature += 1
          splits[@path_pattern % [i[0], i[1]]] << l
        end
      end
    end
    # END YOUR CODE

    return splits
  end

  def test example
    # BEGIN YOUR CODE
    for i in example["features"]
      if i[0] == @feature_name
        return @path_pattern % [i[0], i[1]]
      end
    end
    return nil
    # END YOUR CODE

    return path_name
  end
end

class CategoricalSplitter
  def matches? examples, feature_name
    has_feature = examples.select {|r| r["features"].has_key? feature_name}
    return false if has_feature.empty?
    return has_feature.all? do |r|
      r["features"].fetch(feature_name, 0.0).is_a?(String)
    end
  end

  def create_split examples, parent_entropy, feature_name
    # BEGIN YOUR CODE
    information = []
    if(matches?(examples, feature_name)) == false
      return nil
    end

    cate = CategoricalSplit.new(feature_name)
    split = cate.split_on_feature(examples)
    ig = information_gain(parent_entropy, split)
    # END YOUR CODE

    return {"split" => cate, "information_gain" => ig}
  end
end


class DecisionNode
  attr_reader :children, :examples, :split, :node_entropy, :node_class_distribution

  def initialize examples
    @examples = examples
    @node_class_distribution = class_distribution examples
    @node_entropy = entropy (@node_class_distribution)
    @children = Hash.new
  end

  def is_leaf?
    self.children.empty?
  end

  def score positive_class_label
    # BEGIN YOUR CODE
    if @node_class_distribution.key?(positive_class_label) == false
      return 0.0
    end
    return @node_class_distribution[positive_class_label]
    # END YOUR CODE
  end

  def all_possible_splits feature_names, splitters
    all_splits = []

    # BEGIN YOUR COD
    for i in splitters
      for l in feature_names
        cell = i.create_split(@examples, @node_entropy, l)
        if cell.nil? == false
          if cell["split"].nil? == false or cell["information_gain"] > 0
            all_splits << cell
          end
        end
      end
    end
    # END YOUR CODE

    return all_splits
  end

  def split_node! split
    @split = split
    # BEGIN YOUR CODE
    splited = split.split_on_feature(@examples)
    for i in splited
      @children[i[0]] = DecisionNode.new(i[1])
    end
    # END YOUR CODE

    @examples = nil
  end
end

class DecisionTreeLearner
  include Learner
  attr_reader :root
  attr_accessor :positive_class_label

  def initialize positive_class_label, min_size: 10, max_depth: 50
    @splitters = [CategoricalSplitter.new, NumericSplitter.new]
    @parameters = {"min_size" => min_size, "max_depth" => max_depth}
    @positive_class_label = positive_class_label
  end

  def to_s
    JSON.pretty_generate(summarize_node(@root))
  end

  def summarize_node node
    summary = {
        leaf: node.is_leaf?
    }
    if node.is_leaf?
      summary[:class_distribution] = node.node_class_distribution
    else
      summary[:split] = node.split
      summary[:children] = node.children
      .sort_by{|kv| kv.first}
      .map do |kv|
        path, child = kv
        [path, summarize_node(child)]
      end.to_h
    end

    return summary
  end

  def train dataset
    @feature_names = dataset["features"]
    examples = dataset["data"]
    @root = DecisionNode.new examples
    grow_tree @root, @parameters["max_depth"]
    summarize_node(@root)
  end

  def grow_tree parent, remaining_depth
    # BEGIN YOUR CODE
    depth = remaining_depth - 1
    if (depth <= 0.0) or (parent.examples.size() <= @parameters["min_size"])
      return
    end
    possible = parent.all_possible_splits(@feature_names, @splitters)
    if possible.size <= 0.0
      return
    end
    sorted_splits = possible.sort_by {|split| split["information_gain"]}.reverse
    best_split = sorted_splits.first
    parent.split_node!(best_split["split"])
    for i in parent.children
      grow_tree(i[1], depth)
    end

    # END YOUR CODE
  end

  def predict example
    leaf = find_leaf @root, example
    return leaf.score @positive_class_label
  end

  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] == @positive_class_label ? 1 : 0
      [score, label]
    end
  end

  def find_leaf node, example
    # BEGIN YOUR CODE
    if node.is_leaf? == true or node.split.nil? == true
      return node
    end

    path = node.split.test(example)

    if(node.children[path].nil? == true)
      #print(node.children, " children\n")
      return node
    end

    find_leaf(node.children[path], example)
    # END YOUR CODE
  end
end

class RandomForestLearner
  include Learner
  attr_reader :trees

  def initialize rng, positive_class_label, num_trees: 10, min_size: 10, max_depth: 50
    @rng = rng
    @parameters = {"num_trees" => num_trees, "min_size" => min_size, "max_depth" => max_depth}
    @positive_class_label = positive_class_label
    tree_parameters = @parameters.clone.delete :num_trees

    @trees = Array.new(num_trees) do |i|
      DecisionTreeLearner.new @positive_class_label, min_size: min_size, max_depth: max_depth
    end
  end

  def to_s
    JSON.pretty_generate(@trees.collect {|t| t.summarize_node t.root})
  end

  def random_features_subset dataset
    num_features = 3
    feature_list = dataset["features"].sample(num_features, random: @rng)
  end

  def random_forest_dataset dataset
    feature_list = random_features_subset dataset
    examples = dataset["data"]
    new_dataset = nil

    # BEGIN YOUR CODE
    cloned = dataset.clone()
    clonedExam = examples.clone()
    examlist = []
    for i in 0..clonedExam.size-1
      current = Hash.new
      current["features"] = Hash.new
      store = clonedExam[@rng.rand(clonedExam.size-1)].clone()
      temp = store.clone()
      for l in temp["features"]
        if feature_list.include?(l[0]) == true
          current["features"][l[0]] = store["features"][l[0]].clone()
        end
      end
      for l in temp
        if l[0] != "features"
          current[l[0]] = store[l[0]].clone()
        end
      end
      examlist << current
    end


    cloned["features"] = feature_list
    cloned["data"] = examlist
    # END YOUR CODE
    return cloned
  end

  def train dataset
    # BEGIN YOUR CODE
    for i in @trees
      newData = random_forest_dataset(dataset)
      i.train(newData)
    end
    # END YOUR CODE
  end

  def evaluate eval_dataset
    examples = eval_dataset["data"]
    examples.map do |example|
      score = predict(example)
      label = example["label"] == @positive_class_label ? 1 : 0
      [score, label]
    end
  end

  def predict example
    # BEGIN YOUR CODE
    total = 0.0
    for i in @trees
      total = total + i.predict(example) + 0.0
    end
    return total / (@trees.size() + 0.0)
    # END YOUR CODE
  end
end

