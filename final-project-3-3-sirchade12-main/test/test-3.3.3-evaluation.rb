require 'test/unit'
require 'model-evaluation-test'

class TestModelEvaluation < Test::Unit::TestCase 
	include ModelEvaluation
	class << self
		def startup
			@@context = ModelEvaluation::TestContext.new
			@@context.classifier = ClassifierFour.new

			ModelEvaluation.on_training_startup @@context
		end
	end

	def setup
		@min_auc = 0.73
		@max_auc = 1.0
		@dev_db_size = 15334
		@test_db_size = 15510
		@evaluation_output_filename = "evaluation-3.3.3.json"
		@image_filename = "question_3_2.png"
		puts ""
	end

	def test_question_3_1_run
		can_classifier_run @@context
	end

	def test_question_3_1_cv_performance
		can_check_cv_performance @@context
	end

	def test_question_3_2_dev_set_performance
		can_dev_set_performance @@context
	end

	def test_plot_question_3_2_dev_set_roc
		plot_dev_set_roc @@context
	end

	def test_question_3_3_create_test_set_predictions
		can_create_test_set_predictions @@context
	end
end