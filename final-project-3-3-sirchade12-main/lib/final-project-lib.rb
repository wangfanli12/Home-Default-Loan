require 'json'
require 'distribution'
require 'sqlite3'
require 'timeout'
require 'tty-table'
require 'ml-interfaces'

def load_german_credit_dataset; JSON.parse(File.read(File.join('data', 'german-credit.json'))); end
def load_spambase_dataset; JSON.parse(File.read(File.join('data', 'spambase.json'))); end

def create_dataset db, sql
  examples = []
  feature_names = Hash.new
  
  db.execute sql do |row|
    features = Hash.new      
    unless row.has_key?("TARGET") and row.has_key?("SK_ID_CURR")
        raise ArgumentError.new("Query must include 'target' and 'sk_id_curr'") 
    end
      
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

module FinalProjectClassifier
  ## Perform any SQL queries / transformations you need to do 
  ## Return a dataset to be used to train models
  ## This may be called multiple times during testing
  def create_training_dataset training_db
    
  end
  
  ## Run whatever SQL queries / transformations you need.
  ## Assume this database is different that the training set
  ## For example, if you are doing any sampling on the training set
  ## don't do it here.
  def create_evaluation_dataset evaluation_db
    
  end
  
  ## Return an array of Learners which be evaluated on 5-fold cross-validation
  def create_learners dataset
    
  end
  
  ## Returns predictions in the correct format
  def create_predictions learner, dataset
    dataset["data"].map do |example|
      score = learner.predict example
      [example["id"], score]
    end.to_h
  end
end

def parameter_search learners, dataset, folds = 5
  metric = AUCMetric.new  
  stats = cross_validation_model_performance dataset, folds, learners, metric
  best_by_learner = best_performance_by_learner stats  
    summary = Hash.new
    best_by_learner.each_key do |k|
        summary[k] = best_by_learner[k].clone
        summary[k].delete "trained_model"
    end
  # puts JSON.pretty_generate(summary)

  stats.map! {|s| t = s.clone; t.delete "trained_model"; t}
    
  return [stats, best_by_learner]
end   

def print_stats stats
  header = stats.first.keys

  rows = stats.collect {|r| header.collect do |k| 
    if k == :parameters
      r[k].to_json
    elsif r[k].is_a? Float
      "%.5f" % r[k]
    else
      r[k].to_s
    end
  end
  }

  puts TTY::Table.new(header, rows).render(:ascii)
end

def cross_validation_model_performance dataset, folds, learners, metric    
  learners.map do |learner|
    tr_metrics = []
    te_metrics = []
    puts "#{folds}-fold CV: #{learner.class.name}, parameters: #{learner.parameters}"
    cross_validate dataset, folds do |train_dataset, test_dataset|
      learner.train train_dataset
      train_scores = learner.evaluate train_dataset
      test_scores = learner.evaluate test_dataset      
      tr_metrics << metric.apply(train_scores)
      te_metrics << metric.apply(test_scores)
    end
      
    #Train on full training set
    learner.train dataset
    learner_name = learner.name
    # puts mean(te_metrics)
    {
      "learner" => learner_name, "trained_model" => learner, "parameters" => learner.parameters, "folds" => folds,
      "mean_train_metric" => mean(tr_metrics), "stdev_train_metric" => stdev(tr_metrics),
      "mean_test_metric" => mean(te_metrics), "stdev_test_metric" => stdev(te_metrics),
    }
  end
end

def best_performance_by_learner stats  
  stats.group_by {|s| s["learner"]}.map do |g_s|
    learner, learner_stats = g_s
    best_parameters = learner_stats.max_by {|l| l["mean_test_metric"]}    
    [learner, best_parameters]
  end.to_h
end
