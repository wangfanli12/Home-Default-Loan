require "numo/gnuplot"
require "test/unit"

require 'final-project-lib'
require 'decision_trees'
require 'evaluation'
require 'ml-interfaces'
require 'transforming-learners'

class TestAssignment < Test::Unit::TestCase 
  def setup
    @german_credit = JSON.parse(File.read(File.join("data", 'german-credit.json')))
    @seed = 'eifjcchdivlbcbflbgblfgukbtkhvejvtkevfbtetjnl'.to_i(26)
    @rng = Random.new(@seed)
    puts ""
  end

  def test_question_1_1_dt_german_credit()
    german_credit = load_german_credit_dataset()
    examples = german_credit["data"]
    learner = DecisionTreeLearner.new 1, min_size: 100, max_depth: 10
    learner.train german_credit

    scores = learner.evaluate german_credit
    metric = AUCMetric.new
    fp, tp, training_auc = metric.roc_curve scores
    assert_true(training_auc > 0.75, "AUC Should be 0.75 or more")
  end

  def test_question_1_1_rf_german_credit()
    german_credit = load_german_credit_dataset()
    learner = RandomForestLearner.new @rng, 1, num_trees: 11, min_size: 100, max_depth: 10
    learner.train german_credit
    assert_equal 11, learner.trees.size

    scores = learner.evaluate german_credit
    metric = AUCMetric.new
    fp, tp, training_auc = metric.roc_curve scores
    assert_true(training_auc > 0.8, "AUC Should be 0.8 or more")
  end

  def test_question_1_2
    german_credit = load_german_credit_dataset()
    learners = [
        DecisionTreeLearner.new(1, min_size: 5, max_depth: 50),
        RandomForestLearner.new(@rng, 1, num_trees: 11, min_size: 5, max_depth: 50),
    ]  
    stats, best_model_stats = parameter_search learners, german_credit
    
    assert_true best_model_stats["DecisionTreeLearner"]["mean_test_metric"] > 0.65, "Decision Tree > 0.65"
    assert_true best_model_stats["RandomForestLearner"]["mean_test_metric"] > 0.68, "Random Forest > 0.70"  

    decision_tree_auc = best_model_stats["DecisionTreeLearner"]["mean_test_metric"]
    print_stats stats
  end
end