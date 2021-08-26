require 'test/unit'
require 'model-evaluation-test'

class TestModelEvaluation < Test::Unit::TestCase 
	include ModelEvaluation
	class << self
		def startup
			@@context = ModelEvaluation::TestContext.new
			@@context.classifier = ClassifierFive.new

			ModelEvaluation.on_training_startup @@context
		end
	end

	def setup
		@min_auc = 0.755
		@max_auc = 1.0
		@dev_db_size = 15334
		@test_db_size = 15510
		@evaluation_output_filename = "evaluation-3.3.4.json"
		@image_filename = "question_4_2.png"
		puts ""
	end

	def test_question_4_1_run
		can_classifier_run @@context
	end

	def test_question_4_1_cv_performance
		can_check_cv_performance @@context
	end

	def test_question_4_2_dev_set_performance
		can_dev_set_performance @@context
	end

	def test_plot_question_4_2_dev_set_roc
		plot_dev_set_roc @@context
	end

	def test_question_4_3_create_test_set_predictions
		can_create_test_set_predictions @@context
	end
end