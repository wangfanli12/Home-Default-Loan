# require 'evaluation'

# def plot_roc_curve fp, tp, auc
#   plot = Daru::DataFrame.new({x: fp, y: tp}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
#     plot.x_label "False Positive Rate"
#     plot.y_label "True Positive Rate"
#     diagram.title("AUC: %.4f" % auc)
#     plot.legend(true)
#   end
# end  

# def cross_validation_model_performance dataset, folds, learners, metric    
#   learners.map do |learner|
#     tr_metrics = []
#     te_metrics = []
#     puts "#{folds}-fold CV: #{learner.class.name}, parameters: #{learner.parameters}"
#     cross_validate dataset, folds do |train_dataset, test_dataset|
#       learner.train train_dataset
#       train_scores = learner.evaluate train_dataset
#       test_scores = learner.evaluate test_dataset      
#       tr_metrics << metric.apply(train_scores)
#       te_metrics << metric.apply(test_scores)
#     end
      
#     #Train on full training set
#     learner.train dataset
#     learner_name = learner.name
#     puts mean(te_metrics)
#     {
#       "learner" => learner_name, "trained_model" => learner, "parameters" => learner.parameters, "folds" => folds,
#       "mean_train_metric" => mean(tr_metrics), "stdev_train_metric" => stdev(tr_metrics),
#       "mean_test_metric" => mean(te_metrics), "stdev_test_metric" => stdev(te_metrics),
#     }
#   end
# end

# def best_performance_by_learner stats  
#   stats.group_by {|s| s["learner"]}.map do |g_s|
#     learner, learner_stats = g_s
#     best_parameters = learner_stats.max_by {|l| l["mean_test_metric"]}    
#     [learner, best_parameters]
#   end.to_h
# end

# # def parameter_search learners, dataset, folds = 5
# #   metric = AUCMetric.new  
# #   stats = cross_validation_model_performance dataset, folds, learners, metric
# #   best_by_learner = best_performance_by_learner stats  
# #     summary = Hash.new
# #     best_by_learner.each_key do |k|
# #         summary[k] = best_by_learner[k].clone
# #         summary[k].delete "trained_model"
# #     end
# #   puts JSON.pretty_generate(summary)

# #   assert_equal learners.size, stats.size
# #   assert_true(stats.all? {|s| a = s["mean_train_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
# #   assert_true(stats.all? {|s| a = s["mean_test_metric"]; a >= 0.0 and a <= 1.0}, "0 <= Train AUC <= 1")
  
# #   stats.map! {|s| t = s.clone; t.delete "trained_model"; t}
    
# #   return [stats, best_by_learner]
# # end
