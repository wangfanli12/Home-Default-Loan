# Final Project Part 3.3

Implement three classifiers that achieves an AUC of > 0.7, 0.73, and 0.755 on the final project dataset. This demonstrates that you have put in some work in tuning your models and thinking through some features. It would be helpful to complete the Related Work section of your report while doing this part so see what other work in the literature or blogs inspire you. You will only receive points for a model which performs above each cut point, as measured through a hidden test. No partial credit.

You may choose any classifier discussed during class but only those discussed in class, but you may choose to implement a classifier discussed but not covered in an assignment. You will be creating multiple `FinalProjectClassifier` classes. They are used as follows:

* Call `create_training_dataset` with a database. Use SQL to obtain data, but do not apply any transforms at this time. 
* Call `create_learners` with the dataset. Configures but does not train a learner. Use the `TransformingLearner` to apply feature transformations during the training phase. 
* You may return multiple learners that can be compared using cross validation.
* After obtaining the best learner, call `create_evaluation_dataset`. You may call `apply` on transformations but not `train`. Using the labels in any way on the evaluation set will return in zero credit for this project. 

We will use all the models, feature extraction, and evaluation methods you learned in this course on a realistic dataset. In the final project, you will create better and better classifiers for the final project dataset. In each set of classifiers, create a dataset directly from the database. Assume that the database will be provided to you and **do not assume** that the database will be the same all the time.

Note that the test set performance will be a **HIDDEN** test, so passing the test locally means you can generate a file with predictions. The autograder will evaluate it when you submit. You may find that your model does not perform as well on the hidden test as on the training set. To mitigate this, we are using cross-validation and a hold-out set which is about the same size as the testing set. However, your model's performance is based solely on the hidden tests and no partial credit will be used. 

Each example in the dataset you create will have an ```id``` field. This is how we will track performance. Please note that that you must provide a score for each example in the evaluation database provided to your classifier. So please take care to keep all examples in the evaluation set. You may sample the training set as much as you want. 

Note the following requirements:
1. Focus on data preparation and creating, normalizing, understanding new features.
1. You may use any analysis package you want for exploration, but all implementation of features should be in this repository.
1. Use your own implementation of all code, but you may reuse code from other assignments.
1. Avoid creating new models that we did not cover in the class. There is no credit for fancy models.
1. Do not use the target label, or anything that is derived based on the training label as features. 
1. You may talk to other students about your solution, but do not share code. This is an individual assignment.
1. The final project submissions are manually reviewed and any target leakage (using knowledge from the testing set for training) will result in a severe point reduction.
1. Keep in mind that it is hard to get 100% of these tests correct, but try your best.


It is suggested to validate your model's performance with multiple learners through cross validation on the dev set first and using the full dataset. So, try 5-10 parameter combinations on the dev set, narrow down, and run the full dataset.

## Question 1 
In this part of the final project, you will create your own features and try different combinations of models. This notebook sets up the foundation for the final project. Most of the code you need for the final project has already been implemented by you across multiple assignments. It has been scattered and we have copied and pasted where necessary. 

In this assignment, you will extend your library of the ```Learner```s, ```FeatureTransformer```s and ```Metric```s needed for the final project. Copy / refactor **your** implementation of the ```evaluation.rb``` file located in the `src/` directory. 

where ```<username>``` is your username. You can find that in the URL for this page.

### Question 1.1 (1 points)

Copy / refactor **your** implementations of ```DecisionTreeLearner``` and ```RandomForestLearner``` in the `decision_trees.rb` file.

Check your implementation with the tests below:
```bash
ruby -I src -I lib test/test-integration.rb  -v v --name /test_question_1_1/
```

### Question 1.2 (11 points)

Now, we will test different type models on the same dataset and compare. This is similar to what will happen in the final project, so use this test as an example.

This test should execute in less than 1 minute.

Your table should look like this:

| ID | learner |  parameters | folds | mean_train_metric | stdev_train_metric | mean_test_metric | stdev_test_metric |
|--- | --- |  --- | --- | --- | --- | --- | --- |
|0 | DecisionTreeLearner | {"min_size"=>5, "max_depth"=>50} | 5 | 0.986672584401098 | 0.003775296572533165 | 0.6715093881364055 | 0.03667509113269947 | 
|1 | RandomForestLearner | {"num_trees"=>11, "min_size"=>5, "max_depth"=>50} | 5 | 0.9312726835057689 | 0.008456811357652269 | 0.7024616414815554 | 0.025013141737806444 | 

Check your implementation with the tests below:
```bash
ruby -I src -I lib test/test-integration.rb  -v v --name /test_question_1_2/
```

# Question 2: Classifier 3
Classifier 3 requires an AUC > 0.7. Modify the skeleton class `ClassifierThree` in `final-project-3.3.rb` to customize training and parameters. 

## Question 2.0 (7 points)
Define ```ClassifierThree``` which creates the training set. Assume that the evaluation dataset may come from a different evaluation database so don't just return the training set again.

Validate that the classifier works by trying to create a small dataset and training the model on it. The model will be retrained later. This just verify that the interface is working.

Check your implementation with the test below
```bash
ruby -I src -I lib test/test-3.3.2-setup.rb  -v v --name /test_question_2_0/
```

Notes:
1. This test will timeout after 3 hours, after which you will see an ```Timeout::Error execution expired``` error. If this takes more than 20 minutes, you have probably done something wrong.

## Evaluate your model
Test your model. Before reaching this test, make sure to pass the previous question which runs on a sample dataset. Each test below runs the entire model training over again, which is expensive. Instead of running each test separately, try running all tests below in one pass, which will train your model only once.

```bash
ruby -I src -I lib test/test-3.3.2-evaluation.rb -v v
```

## Question 2.1 (7 points)

Measures the 5-fold cross-validation performance of your classifier. This may take a while, so be cognizant of the fact that your classifier may run out of memory--the server is a shared environment.

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.2-evaluation.rb -v v --name /test_question_2_1/
```

## Question 2.2 (7 points)

Plots the ROC curve on the dev dataset, which is a separate database. Checks that the AUC is within target. Note that this assumes the model has already been trained.

Check your implementation with the test below:
```bash
ruby -I src -I lib test/test-3.3.2-evaluation.rb --name /test_question_2_2/
```

```bash
ruby -I src -I lib test/test-3.3.2-evaluation.rb --name /test_plot_2_2/
```

## Question 2.3 (70 Points)

Tests your model on the test dataset. You will not be able to see the result until after you submit. The evaluation uses a remove model evaluation service to check the predictions. 

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.2-evaluation.rb --name /test_question_2_3/
```

# Question 3: Classifier 4
Classifier 4 requires an AUC > 0.73. Modify the skeleton class `ClassifierFour` in `final-project-3.3.rb` to customize training and parameters. 

## Question 3.0 (7 points)
Define ```ClassifierFour``` which creates the training set. Assume that the evaluation dataset may come from a different evaluation database so don't just return the training set again.

Validate that the classifier works by trying to create a small dataset and training the model on it. The model will be retrained later. This just verify that the interface is working.

Check your implementation with the test below
```bash
ruby -I src -I lib test/test-3.3.3-setup.rb  -v v --name /test_question_3_0/
```

Notes:
1. This test will timeout after 3 hours, after which you will see an ```Timeout::Error execution expired``` error. If this takes more than 20 minutes, you have probably done something wrong.

## Evaluate your model
Test your model. Before reaching this test, make sure to pass the previous question which runs on a sample dataset. Each test below runs the entire model training over again, which is expensive. Instead of running each test separately, try running all tests below in one pass, which will train your model only once.

```bash
ruby -I src -I lib test/test-3.3.3-evaluation.rb -v v
```

## Question 3.1 (7 points)

Measures the 5-fold cross-validation performance of your classifier. This may take a while, so be cognizant of the fact that your classifier may run out of memory--the server is a shared environment.

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.3-evaluation.rb -v v --name /test_question_3_1/
```

## Question 3.2 (7 points)

Plots the ROC curve on the dev dataset, which is a separate database. Checks that the AUC is within target. Note that this assumes the model has already been trained.

Check your implementation with the test below:
```bash
ruby -I src -I lib test/test-3.3.3-evaluation.rb --name /test_question_3_2/
```

```bash
ruby -I src -I lib test/test-3.3.3-evaluation.rb --name /test_plot_3_2/
```

## Question 3.3 (70 Points)

Tests your model on the test dataset. You will not be able to see the result until after you submit. The evaluation uses a remove model evaluation service to check the predictions. 

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.3-evaluation.rb --name /test_question_3_3/
```


# Question 4: Classifier 5
Classifier 5 requires an AUC > 0.755. Modify the skeleton class `ClassifierFive` in `final-project-3.3.rb` to customize training and parameters. 

## Question 4.0 (6 points)
Define ```ClassifierFive``` which creates the training set. Assume that the evaluation dataset may come from a different evaluation database so don't just return the training set again.

Validate that the classifier works by trying to create a small dataset and training the model on it. The model will be retrained later. This just verify that the interface is working.

Check your implementation with the test below
```bash
ruby -I src -I lib test/test-3.3.4-setup.rb  -v v --name /test_question_4_0/
```

Notes:
1. This test will timeout after 3 hours, after which you will see an ```Timeout::Error execution expired``` error. If this takes more than 20 minutes, you have probably done something wrong.

## Evaluate your model
Test your model. Before reaching this test, make sure to pass the previous question which runs on a sample dataset. Each test below runs the entire model training over again, which is expensive. Instead of running each test separately, try running all tests below in one pass, which will train your model only once.

```bash
ruby -I src -I lib test/test-3.3.4-evaluation.rb -v v
```

## Question 4.1 (6 points)

Measures the 5-fold cross-validation performance of your classifier. This may take a while, so be cognizant of the fact that your classifier may run out of memory--the server is a shared environment.

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.4-evaluation.rb -v v --name /test_question_4_1/
```

## Question 4.2 (6 points)

Plots the ROC curve on the dev dataset, which is a separate database. Checks that the AUC is within target. Note that this assumes the model has already been trained.

Check your implementation with the test below:
```bash
ruby -I src -I lib test/test-3.3.4-evaluation.rb --name /test_question_4_2/
```

```bash
ruby -I src -I lib test/test-3.3.4-evaluation.rb --name /test_plot_4_2/
```

## Question 4.3 (60 Points)

Tests your model on the test dataset. You will not be able to see the result until after you submit. The evaluation uses a remove model evaluation service to check the predictions. 

Check your implementation with the test below:

```bash
ruby -I src -I lib test/test-3.3.4-evaluation.rb --name /test_question_4_3/
```




