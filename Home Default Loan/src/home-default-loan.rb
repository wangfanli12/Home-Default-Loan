require 'final-project-lib'
require 'ml-interfaces'
require 'ml-math'
require 'learners'
require 'linear_models'
require 'decision_trees'
require 'transformers'
require 'optimizers'
require 'evaluation'
require 'neural-networks'


class ClassifierThree
  include FinalProjectClassifier

  def initialize
    # BEGIN YOUR CODE
    # END YOUR CODE
  end


  #Extract data from database only, but does not construct features
  def create_original_dataset train_db
    # BEGIN YOUR CODE
    sql = <<SQL
select  application_train.sk_id_curr,  target, ext_source_1, ext_source_2, ext_source_3, name_education_type, days_employed, code_gender, days_last_phone_change, days_birth,
    application_train.amt_annuity, application_train.amt_credit, application_train.amt_goods_price, name_contract_type, name_income_type 
from application_train
order by application_train.sk_id_curr

SQL
    sq2 = <<SQL
select sk_id_curr, days_credit, credit_active
from bureau
order by sk_id_curr
SQL
    dataset = create_dataset train_db, sql
    dataset["data"].map {|x| x["features"]["bias"] = 1.0}
    dataset2 = create_dataset train_db, sq2
    days_credit = Hash.new {|h,k| h[k] = 0.0}
    bad_credit = Hash.new {|h,k| h[k] = 0.0}
    active_credit = Hash.new {|h,k| h[k] = 0.0}
    for i in dataset2["data"]
      days_credit[i["id"]] += i["features"]["days_credit"]
      if i["features"]["credit_active"] == "Bad debt"
        bad_credit[i["id"]] += 1.0
      elsif i["features"]["credit_active"] == "Active"
        active_credit[i["id"]] += 1.0
      end
    end
    dataset["features"].append("days_credit")
    dataset["features"].append("bad_credit")
    dataset["features"].append("active_credit")
    for i in dataset["data"]
      i["features"]["days_credit"] = days_credit[i["id"]]
      i["features"]["bad_credit"] = bad_credit[i["id"]]
      i["features"]["active_credit"] = active_credit[i["id"]]
      amtAnnuity = i["features"].delete("amt_annuity")
      amtCredit = i["features"].delete("amt_credit")
      amtGoodsPrice = i["features"].delete("amt_goods_price")
      daysEmployed = i["features"].delete("days_employed")
      daysBirth = i["features"].delete("days_birth")
      if(amtAnnuity.nil? == false)
        i["features"]["credit_to_annuity_ratio"] = amtCredit/(amtAnnuity + 0.0)
      end
      if(amtGoodsPrice.nil? == false)
        i["features"]["credit_to_goods_ratio"] = amtCredit/(amtGoodsPrice + 0.0)
      end
      if(daysEmployed.nil? == false)
        i["features"]["employed_to_birth_ratio"] = daysEmployed/(daysBirth + 0.0)
      end
    end

    # END YOUR CODE
    return dataset
  end

  def create_training_dataset train_db
    # BEGIN YOUR CODE
    dataset = create_original_dataset(train_db)
    cloned_dataset = clone_dataset(dataset)
    @train = cloned_dataset

    downSampling = DownsampleNegatives.new(0.0)
    downSampling.update_sampling_rate dataset
    downSampling.apply(dataset["data"])

    # END YOUR CODE
    return dataset
  end

  def create_evaluation_dataset evaluation_db
    # BEGIN YOUR CODE
    #cloned = evaluation_db.clone()
    transformer = FeatureTransformPipeline.new(
        #AgeRangeAsVector.new(),
        #TargetAveraging.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        #LogTransform.new(%w(credit_to_annuity_ratio credit_to_goods_ratio)),
        OneHotEncoding.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_annuity_ratio credit_to_goods_ratio days_last_phone_change employed_to_birth_ratio
         days_credit bad_credit active_credit)),
        MeanImputation.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_goods_ratio credit_to_annuity_ratio))
    )

    dataset = create_original_dataset(evaluation_db)

    transformer.train(@train)
    transformer.apply dataset["data"]

    # END YOUR CODE
    return dataset
  end

  def create_learners dataset
    @rng = Random.new(358430676781)
    # BEGIN YOUR CODE
    learners = []
    # BEGIN YOUR CODE


    @transformer2 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        #AgeRangeAsVector.new(),
        #TargetAveraging.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        #LogTransform.new(%w(credit_to_annuity_ratio credit_to_goods_ratio)),
        OneHotEncoding.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_annuity_ratio credit_to_goods_ratio days_last_phone_change employed_to_birth_ratio
        days_credit bad_credit active_credit)),
        MeanImputation.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_goods_ratio credit_to_annuity_ratio))
    )

    @transformer3 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        #DownsampleNegatives.new(0.3),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 amt_annuity amt_credit amt_goods_price))
    )

    @transformer4 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 amt_annuity amt_credit amt_goods_price))
        #downSampling
    )


    #linear1 = RandomForestLearner.new @rng, 1, num_trees: 5, min_size: 200, max_depth: 5
    #problem1 = create_problem(0.3, 358430676781, [5])
    #linear1 = NeuralNetworkLearner.new(batch_size: 32, epochs: 100, problem: problem1)
    #linear1 = DecisionTreeLearner.new(positive_class_label: 1)
    #print(problem1["weights"], " weights for problem1\n")
    #problem2 = create_problem(0.7, 358430676781, [6])
    #linear2 = NeuralNetworkLearner.new(batch_size: 32, epochs: 50, problem: problem2)
    #print(problem2["weights"], " weights for problem2\n")


    linear1 = LogisticRegressionLearner.new(regularization: 0.5, learning_rate: 0.3, batch_size: 64, epochs: 200)
    linear2 = LogisticRegressionLearner.new(regularization: 1.0, learning_rate: 0.01, batch_size: 64, epochs: 200)

    learner1 = CopyingTransformingLearner.new(@transformer2, linear1)
    learner1.name = "1"
    learner2 = CopyingTransformingLearner.new(@transformer3, linear2)
    learner2.name = "2"
    learners << learner1
    learners << learner2
    return learners
    # END YOUR CODE
  end

  def clone_example example
    e = example.clone
    e["features"] = example["features"].clone
    return e
  end

  def clone_dataset dataset
    cloned_dataset = dataset.clone
    cloned_dataset["features"] = dataset["features"].clone
    cloned_dataset["data"] = dataset["data"].map {|e| clone_example(e)}
    return cloned_dataset
  end
end

class ClassifierFour
  include FinalProjectClassifier


  def initialize
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE
  end

  #Extract data from database only, but does not construct features
  def create_original_dataset train_db
    # BEGIN YOUR CODE
    sql = <<SQL
select  application_train.sk_id_curr,  target, ext_source_1, ext_source_2, ext_source_3, name_education_type, days_employed, code_gender, days_last_phone_change, days_birth,
    application_train.amt_annuity, application_train.amt_credit, application_train.amt_goods_price, name_contract_type, name_income_type
from application_train
order by application_train.sk_id_curr

SQL
    sq2 = <<SQL
select sk_id_curr, days_credit, credit_active
from bureau
order by sk_id_curr
SQL
    dataset = create_dataset train_db, sql
    dataset["data"].map {|x| x["features"]["bias"] = 1.0}
    dataset2 = create_dataset train_db, sq2
    days_credit = Hash.new {|h,k| h[k] = 0.0}
    bad_credit = Hash.new {|h,k| h[k] = 0.0}
    active_credit = Hash.new {|h,k| h[k] = 0.0}
    for i in dataset2["data"]
      days_credit[i["id"]] += i["features"]["days_credit"]
      if i["features"]["credit_active"] == "Bad debt"
        bad_credit[i["id"]] += 1.0
      elsif i["features"]["credit_active"] == "Active"
        active_credit[i["id"]] += 1.0
      end
    end
    dataset["features"].append("days_credit")
    dataset["features"].append("bad_credit")
    dataset["features"].append("active_credit")
    for i in dataset["data"]
      i["features"]["days_credit"] = days_credit[i["id"]]
      i["features"]["bad_credit"] = bad_credit[i["id"]]
      i["features"]["active_credit"] = active_credit[i["id"]]
      amtAnnuity = i["features"].delete("amt_annuity")
      amtCredit = i["features"].delete("amt_credit")
      amtGoodsPrice = i["features"].delete("amt_goods_price")
      daysEmployed = i["features"].delete("days_employed")
      daysBirth = i["features"].delete("days_birth")
      if(amtAnnuity.nil? == false)
        i["features"]["credit_to_annuity_ratio"] = amtCredit/(amtAnnuity + 0.0)
      end
      if(amtGoodsPrice.nil? == false)
        i["features"]["credit_to_goods_ratio"] = amtCredit/(amtGoodsPrice + 0.0)
      end
      if(daysEmployed.nil? == false)
        i["features"]["employed_to_birth_ratio"] = daysEmployed/(daysBirth + 0.0)
      end
    end

    # END YOUR CODE
    return dataset
  end

  def create_training_dataset train_db
    # BEGIN YOUR CODE
    dataset = create_original_dataset(train_db)
    cloned_dataset = clone_dataset(dataset)
    @train = cloned_dataset

    downSampling = DownsampleNegatives.new(0.0)
    downSampling.update_sampling_rate dataset
    downSampling.apply(dataset["data"])

    # END YOUR CODE
    return dataset
  end

  def create_evaluation_dataset evaluation_db
    # BEGIN YOUR CODE
    #cloned = evaluation_db.clone()
    transformer = FeatureTransformPipeline.new(
        #AgeRangeAsVector.new(),
        #TargetAveraging.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        #LogTransform.new(%w(credit_to_annuity_ratio credit_to_goods_ratio)),
        OneHotEncoding.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_annuity_ratio credit_to_goods_ratio days_last_phone_change employed_to_birth_ratio
         days_credit bad_credit active_credit)),
        MeanImputation.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_goods_ratio credit_to_annuity_ratio))
    )

    dataset = create_original_dataset(evaluation_db)

    transformer.train(@train)
    transformer.apply dataset["data"]

    # END YOUR CODE
    return dataset
  end

  def create_learners dataset
    @rng = Random.new(358430676781)
    # BEGIN YOUR CODE
    learners = []
    # BEGIN YOUR CODE


    @transformer2 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        #AgeRangeAsVector.new(),
        #TargetAveraging.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        #LogTransform.new(%w(credit_to_annuity_ratio credit_to_goods_ratio)),
        OneHotEncoding.new(%w(name_education_type code_gender name_contract_type name_income_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_annuity_ratio credit_to_goods_ratio days_last_phone_change employed_to_birth_ratio
        days_credit bad_credit active_credit)),
        MeanImputation.new(%w(ext_source_1 ext_source_2 ext_source_3 credit_to_goods_ratio credit_to_annuity_ratio))
    )

    @transformer3 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        #DownsampleNegatives.new(0.3),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 amt_annuity amt_credit amt_goods_price))
    )

    @transformer4 = FeatureTransformPipeline.new(
        #OneHotEncoding.new(%w(name_education_type)),
        ZScoreTransformer.new(%w(ext_source_1 ext_source_2 ext_source_3 amt_annuity amt_credit amt_goods_price))
        #downSampling
    )


    #linear1 = RandomForestLearner.new @rng, 1, num_trees: 5, min_size: 200, max_depth: 5
    #problem1 = create_problem(0.3, 358430676781, [5])
    #linear1 = NeuralNetworkLearner.new(batch_size: 32, epochs: 100, problem: problem1)
    #linear1 = DecisionTreeLearner.new(positive_class_label: 1)
    #print(problem1["weights"], " weights for problem1\n")
    #problem2 = create_problem(0.7, 358430676781, [6])
    #linear2 = NeuralNetworkLearner.new(batch_size: 32, epochs: 50, problem: problem2)
    #print(problem2["weights"], " weights for problem2\n")


    linear1 = LogisticRegressionLearner.new(regularization: 0.5, learning_rate: 0.3, batch_size: 64, epochs: 200)
    linear2 = LogisticRegressionLearner.new(regularization: 1.0, learning_rate: 0.01, batch_size: 64, epochs: 200)

    learner1 = CopyingTransformingLearner.new(@transformer2, linear1)
    learner1.name = "1"
    learner2 = CopyingTransformingLearner.new(@transformer3, linear2)
    learner2.name = "2"
    learners << learner1
    learners << learner2
    # END YOUR CODE
  end

  def clone_example example
    e = example.clone
    e["features"] = example["features"].clone
    return e
  end

  def clone_dataset dataset
    cloned_dataset = dataset.clone
    cloned_dataset["features"] = dataset["features"].clone
    cloned_dataset["data"] = dataset["data"].map {|e| clone_example(e)}
    return cloned_dataset
  end
end

class ClassifierFive
  include FinalProjectClassifier


  def initialize
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE
  end

  #Extract data from database only, but does not construct features
  def create_original_dataset train_db
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE
    return dataset
  end

  def create_training_dataset train_db
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE
    return dataset
  end

  def create_evaluation_dataset evaluation_db
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE
    return dataset

  end

  def create_learners dataset
    @rng = Random.new(358430676781)
    # BEGIN YOUR CODE
    raise NotImplementedError.new('Replace this with your code')
    # END YOUR CODE

    return learners
  end
end

#a version of creating dataset that doesn't have limitation on target and sk_id_curr
def create_dataset db, sql
  examples = []
  feature_names = Hash.new

  db.execute sql do |row|
    features = Hash.new
    fields = row.keys.select {|k| k.is_a? String}.map{|k| k.downcase} - ["target", "sk_id_curr"]
    fields.each do |k|
      v = row[k.upcase]
      next if v.is_a? String and v == ""
      features[k] = v
    end
    fields.each {|k| feature_names[k] = 1}

    u = {"label" => row["TARGET"], "id" => row["SK_ID_CURR"]}
    u["features"] = features
    examples << u
  end
  dataset = {
      "features" => feature_names.keys,
      "data" => examples,
  }
  return dataset
end
