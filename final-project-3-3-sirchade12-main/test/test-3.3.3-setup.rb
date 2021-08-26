require 'test/unit'
require 'model-evaluation-test'

class TestModelSetup < Test::Unit::TestCase 
	#Tests are defined in the module below
	include ModelEvaluation
	
	def setup
		@context = ModelEvaluation::TestContext.new
		@context.classifier = ClassifierFour.new			
		ModelEvaluation.on_setup @context	
		@test = self	
		@dev_db_size = 15334
		puts ""
	end

	def test_question_3_0_defines_classifier
		defines_classifier @context
	end

	def test_question_3_0_create_training_dataset	  
		can_create_training_dataset @context
	end

	def test_question_3_0_create_small_sample
		can_create_small_sample @context
	end

	def test_question_3_0_create_learner
		can_create_learner @context		
	end

	def test_question_3_0_evaluate_learner
		can_evaluate_learner @context		
	end

	def test_question_3_0_create_evaluation_set
		can_create_evaluation_set @context		
	end

	def test_question_3_0_run_on_dev_set
		can_run_on_dev_set @context		
	end	
end