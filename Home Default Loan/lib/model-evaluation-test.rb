require 'test/unit'
require 'final-project-3.3'
require 'evaluation.rb'
require "numo/gnuplot"

module ModelEvaluation
	def ModelEvaluation.data_dir
	  return File.join("..","datasets")
	end

	def ModelEvaluation.load_db db_name
	  #Initializes the database used for this assignment    
	  dir = data_dir()
	  dev_db = SQLite3::Database.new File.join(dir, db_name), results_as_hash: true, readonly: true
	  dev_db.execute "pragma temp_store = 1;"
	  dev_db.execute "pragma temp_store_directory = '#{dir}';"

	  return dev_db
	end

	class TestContext
		attr_accessor :classifier, :train_db, :dev_db, :test_db, :learners, :summary, :cross_validation_results
	end

	def ModelEvaluation.on_setup context	
		context.train_db = load_db "credit_risk_data_train.db"
		context.dev_db = load_db "credit_risk_data_dev.db"
		context.test_db = load_db "credit_risk_data_test.db"
	end


	def ModelEvaluation.on_training_startup context
		folds = 5
		context.train_db = load_db "credit_risk_data_train.db"
		context.dev_db = load_db "credit_risk_data_dev.db"
		context.test_db = load_db "credit_risk_data_test.db"

		puts "Creating training dataset"
		training_set = context.classifier.create_training_dataset context.train_db

		puts "Creating learners"
		context.learners = context.classifier.create_learners training_set

		# Summary contains the cross-validation results
	  puts "Running #{folds}-fold cross validation"
	  context.summary = nil
	  context.cross_validation_results = nil

	  Timeout::timeout(10800) do
	      stats, summary = parameter_search context.learners, training_set, folds
	      context.summary = summary
	      context.cross_validation_results = stats  
	  end
	  # p "Result of #{folds}-fold cross validation run", context.summary
		print_stats context.cross_validation_results
	  
	  # p "Best model result is: ", context.cross_validation_results
	  puts "----- Beginning model evaluation -----"
	end

	def defines_classifier context
		assert_not_nil context
		assert_true context.classifier.class.include?(FinalProjectClassifier)
		assert_not_nil context.classifier				
	end

	def can_create_training_dataset context
		dev_training_set = context.classifier.create_training_dataset context.dev_db
	  assert_not_nil dev_training_set
	  assert_not_nil dev_training_set["features"]
	  assert_false dev_training_set["features"].empty? 
	  assert_not_nil dev_training_set["data"]
		assert_false dev_training_set["data"].empty?

		first_example = dev_training_set["data"][0]
		puts "Dev set features: ", dev_training_set["features"], 
			"First example: ", first_example

	  assert_true(dev_training_set["data"].size > 1, "> 1 examples on dev training set")
	  assert_false(first_example["features"].empty?)
	  assert_false(first_example["label"].nil?)
	end

	def can_create_small_sample context
		dev_training_set = context.classifier.create_training_dataset context.dev_db
		small_training_set = dev_training_set.clone
	  small_training_set["data"] = small_training_set["data"].sample(100)

	  assert_equal 100, small_training_set["data"].size
	end

	def can_create_evaluation_set context
		dev_training_set = context.classifier.create_training_dataset context.dev_db
		small_training_set = dev_training_set.clone
	  small_training_set["data"] = small_training_set["data"].sample(100)

	  assert_equal 100, small_training_set["data"].size
		eval_training_set = context.classifier.create_evaluation_dataset context.dev_db
		assert_equal @dev_db_size, eval_training_set["data"].size
	end

	def can_create_learner context
		dev_training_set = context.classifier.create_training_dataset context.dev_db
		small_training_set = dev_training_set.clone
	  small_training_set["data"] = small_training_set["data"].sample(100)

	  dev_learners = context.classifier.create_learners small_training_set
	  assert_not_nil dev_learners
	  assert_false dev_learners.empty?, "At least 1 learner"

	  assert_true(dev_learners.all? {|l| l.class.include?(Learner)})
	end

	def can_evaluate_learner context
		dev_training_set = context.classifier.create_training_dataset context.dev_db
		small_training_set = dev_training_set.clone
	  small_training_set["data"] = small_training_set["data"].sample(100)

	  dev_learners = context.classifier.create_learners small_training_set
	  dev_learner = dev_learners.first
  	dev_learner.train small_training_set

  	puts "\nModel trained on dev set:", dev_learner.name
	  test_example = small_training_set["data"][0]
	  test_example_score = dev_learner.predict(test_example)

	  puts "\nTesting on", test_example, "Score: ", test_example_score
	  assert_not_nil test_example_score
	  assert_false test_example_score.zero?
	end

	def can_classifier_run context
	  assert_not_nil context.summary
	  assert_not_nil context.cross_validation_results
	  assert_false context.cross_validation_results.empty?
	  assert_equal context.learners.size, context.cross_validation_results.size
	  assert_true(context.cross_validation_results.all? {|s| a = s["mean_train_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
	  assert_true(context.cross_validation_results.all? {|s| a = s["mean_test_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
	  
	  first_summary = context.cross_validation_results.first
	  assert_true(first_summary["stdev_train_metric"] > 0)
	end

	def can_run_on_dev_set context
		folds = 5
		context.train_db = ModelEvaluation.load_db "credit_risk_data_train.db"
		context.dev_db = ModelEvaluation.load_db "credit_risk_data_dev.db"
		context.test_db = ModelEvaluation.load_db "credit_risk_data_test.db"

		puts "Creating training dataset"
		training_set = context.classifier.create_training_dataset context.dev_db

		puts "Creating learners"
		context.learners = context.classifier.create_learners training_set

		# Summary contains the cross-validation results
	  puts "Running #{folds}-fold cross validation"
	  context.summary = nil
	  context.cross_validation_results = nil

	  Timeout::timeout(10800) do
	      stats, summary = parameter_search context.learners, training_set, folds
	      context.summary = summary
	      context.cross_validation_results = stats  
	  end

	  # p "Result of #{folds}-fold cross validation run", context.summary
	  print_stats context.cross_validation_results

	  p "Best model result is: ", context.cross_validation_results
	  puts "----- Beginning model evaluation -----"
	  assert_not_nil context.cross_validation_results
	  assert_not_nil context.summary
	end

	def can_check_cv_performance context
    # Best learner summary contains the cross-validation results for the best learner
	  best_learner_name = context.summary.keys.max_by {|k| context.summary[k]["mean_test_metric"]}  
	  best_learner_summary = context.summary[best_learner_name]
	  
	  # Best learner contains a learner trained on the FULL training set"
	  best_learner = best_learner_summary["trained_model"]  
	  
	  cv_auc = best_learner_summary["mean_test_metric"]
	  puts "Best Learner", best_learner_name, "AUC = #{cv_auc}"
	  assert_true(cv_auc > @min_auc, "Cross-Validation AUC #{cv_auc} > #{@min_auc}")
	  assert_true(cv_auc <= @max_auc, "Cross-Validation AUC #{cv_auc} <= #{@max_auc}") 
	end

	def get_labels_for db, predictions
	  ids = predictions.keys.join(", ")
	  sql = "select sk_id_curr, target from application_train"
	  scores = Array.new
	  db.execute(sql) do |row|
	      id = row["SK_ID_CURR"].to_i
	      unless predictions.has_key? id
	          raise ArgumentError.new("There is no prediction for #{id}. Make sure you are not removing any records.") 
	      end
	    y_hat = predictions[id]
	    y = row["TARGET"]
	    scores << [y_hat, y]
	  end
	  return scores
	end

	def can_dev_set_performance context
	  puts "Creating evaluation dataset"
	  eval_dataset = context.classifier.create_evaluation_dataset context.dev_db
	  puts "Evaluating classifier"
	  best_learner_name = context.summary.keys.max_by {|k| context.summary[k]["mean_test_metric"]}  
	  best_learner_summary = context.summary[best_learner_name]
	  
	  # Best learner contains a learner trained on the FULL training set"
	  best_learner = best_learner_summary["trained_model"]  
	  
	  predictions = context.classifier.create_predictions best_learner, eval_dataset
	  puts predictions.entries[0,5]
	  
	  #Scores on evaluation database
	  puts "Validating predictions against labels from database"
	  scores_on_evaluation_set = get_labels_for context.dev_db, predictions
	  assert_equal @dev_db_size, scores_on_evaluation_set.size, "Returns a score for every example in evaluation set"
	  
	  metric = AUCMetric.new if metric.nil?
	  fp, tp, auc = metric.roc_curve scores_on_evaluation_set
	  puts "Dev set AUC: #{auc}"

	  assert_equal(@dev_db_size + 1, fp.size, "Get all the points")
	  assert_true(auc > @min_auc, "Dev set AUC: #{auc} > #{@min_auc}")
	  assert_true(auc <= @max_auc, "Dev set AUC: #{auc} <= #{@max_auc}")
	end

	def can_create_test_set_predictions context
	  test_dataset = context.classifier.create_evaluation_dataset context.test_db
	  best_learner_name = context.summary.keys.max_by {|k| context.summary[k]["mean_test_metric"]}  
	  best_learner_summary = context.summary[best_learner_name]
	  
	  # Best learner contains a learner trained on the FULL training set"
	  best_learner = best_learner_summary["trained_model"]  
	  
	  predictions = context.classifier.create_predictions best_learner, test_dataset
	  
	  assert_equal @test_db_size, predictions.size, "Returns a score for every example in evaluation set"
	  
	  puts "First 10 prediction scores"
	  p (predictions.keys[0,10].collect {|k| [k,predictions[k]]}.to_h)

	  File.open(@evaluation_output_filename, "w") {|out| out.puts JSON.pretty_generate(predictions)}

		assert_true(File.exists?(@evaluation_output_filename))
		assert_false(File.empty?(@evaluation_output_filename))
	end
	
	def plot_dev_set_roc context
	  eval_dataset = context.classifier.create_evaluation_dataset context.dev_db
	  best_learner_name = context.summary.keys.max_by {|k| context.summary[k]["mean_test_metric"]}  
	  best_learner_summary = context.summary[best_learner_name]
	  
	  # Best learner contains a learner trained on the FULL training set"
	  best_learner = best_learner_summary["trained_model"]  
	  
	  predictions = context.classifier.create_predictions best_learner, eval_dataset
	  
	  #Scores on evaluation database
	  scores_on_evaluation_set = get_labels_for context.dev_db, predictions
	  assert_equal @dev_db_size, scores_on_evaluation_set.size, "Returns a score for every example in evaluation set"
	  
	  puts "Plotting ROC curve"
	  metric = AUCMetric.new if metric.nil?
	  fp, tp, auc = metric.roc_curve scores_on_evaluation_set
	  
	  auc = "%.4f" % auc 
		gp = Numo::Gnuplot.new
    gp.set title:"Dev set ROC Curve (#{auc})"
    gp.set terminal: "png"
    gp.set output: @image_filename
    gp.set xlabel: "False Positive Rate"
    gp.set ylabel: "True Positive Rate"

    gp.plot [fp, tp, w:"lines", notitle:true], [[0,1], [0,1], w:"lines", dashtype: 3, notitle: true]
	end	
end