from sklearn import svm
from Preprocessing import preprocess
from Report_Results import report_results
import numpy as np
from utils import *
import Postprocessing
from datetime import datetime
from sklearn.ensemble import AdaBoostRegressor

metrics = ["sex", "age_cat", 'race', 'c_charge_degree', 'priors_count']
training_data, training_labels, test_data, test_labels, categories, mappings = preprocess(metrics, recalculate=False, causal=False)
np.random.seed(42)
#SVR = svm.LinearSVR(C=1.0 / float(len(test_data)), max_iter=5000)
SVR=AdaBoostRegressor(random_state=0, n_estimators=100)
SVR.fit(training_data, training_labels)

training_predictions = SVR.predict(training_data)
test_predictions = SVR.predict(test_data)

training_race_cases = get_cases_by_metric(training_data, categories, "race", mappings, training_predictions, training_labels)
test_race_cases = get_cases_by_metric(test_data, categories, "race", mappings, test_predictions, test_labels)

training_race_cases, thresholds = Postprocessing.enforce_demographic_parity(training_race_cases, 0.02)

testing_race_cases={}

for group in test_race_cases.keys():
   test_race_cases[group] = apply_threshold(test_race_cases[group], thresholds[group])

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