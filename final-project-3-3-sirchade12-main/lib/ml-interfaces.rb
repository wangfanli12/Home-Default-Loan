module Learner  
	attr_reader :parameters
	def name
			self.class.name
	end

	def train train_dataset    

	end

	def predict example

	end

	def evaluate eval_dataset

	end
end

module Metric
	def name
			self.class.name
	end

	def apply scores
	end
end

module FeatureTransformer
	def name
			self.class.name
	end
		
	def train dataset
			## Calculate any statistics
	end

	def apply example_batch
			## Apply transform to a batch of examples
	end
end
