from sklearn.naive_bayes import MultinomialNB

import Postprocessing
from Preprocessing import preprocess
# from Postprocessing import *
from utils import *

metrics = ["race", "sex", "age", 'c_charge_degree', 'priors_count', 'c_charge_desc']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics)

NBC = MultinomialNB()
NBC.fit(training_data, training_labels)

training_class_predictions = NBC.predict_proba(training_data)
training_predictions = []
test_class_predictions = NBC.predict_proba(test_data)
test_predictions = []

for i in range(len(training_labels)):
    training_predictions.append(training_class_predictions[i][1])

for i in range(len(test_labels)):
    test_predictions.append(test_class_predictions[i][1])

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

# training_race_cases, thresholds = Postprocessing.enforce_predictive_parity(training_race_cases, 0.01)

training_race_cases, thresholds = Postprocessing.enforce_demographic_parity(training_race_cases, 0.02)

for group in test_race_cases.keys():
   test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])



# ADD MORE PRINT LINES HERE - THIS ALONE ISN'T ENOUGH
# YOU NEED ACCURACY AND COST FOR TRAINING AND TEST DATA
# PLUS WHATEVER RELEVANT METRICS ARE USED IN YOUR POSTPROCESSING METHOD, TO ENSURE EPSILON WAS ENFORCED
print("Accuracy on training data:")
print(get_total_accuracy(training_race_cases))
print("")

print("Cost on training data:")
print('${:,.0f}'.format(apply_financials(training_race_cases)))
print("")

print("Accuracy of test data:")
print(get_total_accuracy(test_race_cases))

print("Cost on test data:")
print('${:,.0f}'.format(apply_financials(test_race_cases)))
print("")