
#######################################################################################################################
# YOU MUST FILL OUT YOUR SECONDARY OPTIMIZATION METRIC (either accuracy or cost)!
# The metric chosen must be the same for all 5 methods.
#
# Chosen Secondary Optimization Metric: #
#######################################################################################################################
""" Determines the thresholds such that each group has equal predictive positive rates within 
    a tolerance value epsilon. For the Naive Bayes Classifier and SVM you should be able to find
    a nontrivial solution with epsilon=0.02. 
    Chooses the best solution of those that satisfy this constraint based on chosen 
    secondary optimization criteria.
"""
import matplotlib.pyplot as plt
import utils
import copy
from utils import *
import numpy as np
from datetime import datetime


def enforce_demographic_parity(categorical_results, epsilon):
    demographic_parity_data = {}
    thresholds = {}
    demographic_parity_data = {}
    initial_pprvalues = [[] for x in range(len(categorical_results.keys()))]
    threshold_step = 0.001 #0.01
    ppr_step = 250 #250

    for i in np.arange(0, 1, threshold_step):
        j = 0
        for group in categorical_results.keys():
            demographic_parity_data[group] = utils.apply_threshold(categorical_results[group], i)
            initial_pprvalues[j].append(utils.get_num_predicted_positives(demographic_parity_data[group])/len(demographic_parity_data[group]))
            j = j + 1
    max_ppr = []
    min_ppr = []
    for i in range(len(initial_pprvalues)):
        max_ppr.append(max(initial_pprvalues[i]))
        min_ppr.append(min(initial_pprvalues[i]))

    lower_limit = max(min_ppr)
    upper_limit = min(max_ppr)

    xaxis = np.arange(0, 1, threshold_step)
    zipped_sorted_ppr = []
    arrays1 = []
    sorted_ppr = [[] for x in range(len(categorical_results.keys()))]
    sorted_thresholds = [[] for x in range(len(categorical_results.keys()))]
    for i in range(0, len(categorical_results.keys())):
        arrays = zip(initial_pprvalues[i], xaxis)
        zipped_sorted_ppr = sorted(arrays)
        tuples = zip(*zipped_sorted_ppr)
        temp1, temp2 = [list(tuple) for tuple in tuples]
        sorted_ppr[i] = temp1
        sorted_thresholds[i] = temp2

    steps = (upper_limit - lower_limit) / ppr_step
    k = 0
    i_append = []
    thresoldsAtpprs = [[] for x in range(len(categorical_results.keys()))]
    for i in np.arange(lower_limit, upper_limit+steps, steps):
        k = k + 1
        for j in range(0, len(categorical_results.keys())):
            thresoldsAtpprs[j].append(np.interp(i, sorted_ppr[j], sorted_thresholds[j]))
        i_append.append(i)

    final_Ep_thresholds = []
    thresoldsAtpprs = np.array(thresoldsAtpprs)
    t = thresoldsAtpprs.T
    final_pprvalues = [[] for x in range(t.shape[0])]
    final_Ep_pprvalues = []
    for i in range(0, t.shape[0]):
        temp = 0
        for group in categorical_results.keys():
            demographic_parity_data[group] = utils.apply_threshold(categorical_results[group], t[i][temp])
            temp = temp + 1
            final_pprvalues[i].append(utils.get_num_predicted_positives(demographic_parity_data[group])/len(demographic_parity_data[group]))
        if (max(final_pprvalues[i]) - min(final_pprvalues[i]) <= epsilon):
            final_Ep_pprvalues.append(final_pprvalues[i])
            final_Ep_thresholds.append(t[i])

    max_accuracy = 0
    max_profit = -999999999999999
    total_accuracy_at_maxProfit=0
    final_thresholds_cost=[]
    for i in range(0, len(final_Ep_thresholds)):
        temp = 0
        for group in categorical_results.keys():
            demographic_parity_data[group] = utils.apply_threshold(categorical_results[group],
                                                                  final_Ep_thresholds[i][temp])
            temp = temp + 1

        total_accuracy = utils.get_total_accuracy(demographic_parity_data)
        if (total_accuracy > max_accuracy):
            max_accuracy = total_accuracy
            final_thresholds = final_Ep_thresholds[i]
            total_cost_at_maxacc = utils.apply_financials(demographic_parity_data)
        total_cost = utils.apply_financials(demographic_parity_data)
        if (total_cost > max_profit) :
            max_profit = total_cost
            final_thresholds_cost = final_Ep_thresholds[i]
            total_accuracy_at_maxProfit = utils.get_total_accuracy(demographic_parity_data)
            final_demographic_parity_data = copy.deepcopy(demographic_parity_data)

    i = 0
    for group in categorical_results.keys():
        thresholds[group] = final_thresholds_cost[i]
        i = i + 1

    # print("Demographic parity Maximum accuracy is ",max_accuracy, "at thresholds",final_thresholds, ". Cost is",total_cost_at_maxacc)
    # print("Demographic parity Maximum Profit is :",max_profit)
    # print("At thresholds :",thresholds)
    # print("Respective Accuracy is :",total_accuracy_at_maxProfit)
    #
    # print("--------------------DEMOGRAPHIC PARITY RESULTS--------------------")
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     num_positive_predictions = utils.get_num_predicted_positives(final_demographic_parity_data[group])
    #     prob = num_positive_predictions / len(demographic_parity_data[group])
    #     print("Probability of positive prediction for " + str(group) + ": " + str(prob))
    #
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     accuracy = utils.get_num_correct(final_demographic_parity_data[group]) / len(final_demographic_parity_data[group])
    #     print("Accuracy for " + group + ": " + str(accuracy))
    #
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     FPR = utils.get_false_positive_rate(final_demographic_parity_data[group])
    #     print("FPR for " + group + ": " + str(FPR))
    #
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     FNR = utils.get_false_negative_rate(final_demographic_parity_data[group])
    #     print("FNR for " + group + ": " + str(FNR))
    #
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     TPR = utils.get_true_positive_rate(final_demographic_parity_data[group])
    #     print("TPR for " + group + ": " + str(TPR))
    #
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     TNR = utils.get_true_negative_rate(final_demographic_parity_data[group])
    #     print("TNR for " + group + ": " + str(TNR))
    #
    # print("")
    # for group in thresholds.keys():
    #     print("Threshold for " + group + ": " + str(thresholds[group]))
    #
    # # Must complete this function!
    # print("")
    # for group in final_demographic_parity_data.keys():
    #     fscore_temp=utils.calculate_Fscore(final_demographic_parity_data[group])
    #     print("Fscore for", group, fscore_temp)
    return final_demographic_parity_data, thresholds
    # return None, None

#######################################################################################################################
""" Determine thresholds such that all groups have equal TPR within some tolerance value epsilon, 
    and chooses best solution according to chosen secondary optimization criteria. For the Naive 
    Bayes Classifier and SVM you should be able to find a non-trivial solution with epsilon=0.01
"""

def enforce_equal_opportunity(categorical_results, epsilon):
    thresholds = {}
    equal_opportunity_data = {}
    initial_tprvalues = [[] for x in range(len(categorical_results.keys()))]
    threshold_step = 0.001 #0.001
    tpr_step = 650 #650

    for i in np.arange(0, 1, threshold_step):
        j = 0
        for group in categorical_results.keys():
            equal_opportunity_data[group] = utils.apply_threshold(categorical_results[group], i)
            initial_tprvalues[j].append(utils.get_true_positive_rate(equal_opportunity_data[group]))
            j = j + 1

    max_tpr = []
    min_tpr = []
    for i in range(len(initial_tprvalues)):
        max_tpr.append(max(initial_tprvalues[i]))
        min_tpr.append(min(initial_tprvalues[i]))

    lower_limit = max(min_tpr)
    upper_limit = min(max_tpr)

    xaxis = np.arange(0, 1, threshold_step)
    zipped_sorted_tpr = []
    arrays1 = []
    sorted_tpr = [[] for x in range(len(categorical_results.keys()))]
    sorted_thresholds = [[] for x in range(len(categorical_results.keys()))]
    for i in range(0, len(categorical_results.keys())):
        arrays = zip(initial_tprvalues[i], xaxis)
        zipped_sorted_tpr = sorted(arrays)
        # arrays1.append(zipped_sorted_tpr)
        tuples = zip(*zipped_sorted_tpr)
        temp1, temp2 = [list(tuple) for tuple in tuples]
        sorted_tpr[i] = temp1
        sorted_thresholds[i] = temp2

    steps = (upper_limit - lower_limit) / tpr_step
    k = 0
    i_append = []
    thresoldsAtTPRs = [[] for x in range(len(categorical_results.keys()))]
    for i in np.arange(lower_limit, upper_limit, steps):
        k = k + 1
        for j in range(0, len(categorical_results.keys())):
            thresoldsAtTPRs[j].append(np.interp(i, sorted_tpr[j], sorted_thresholds[j]))
        i_append.append(i)

    final_Ep_thresholds = []
    thresoldsAtTPRs = np.array(thresoldsAtTPRs)
    t = thresoldsAtTPRs.T
    final_tprvalues = [[] for x in range(t.shape[0])]
    final_Ep_tprvalues = []
    for i in range(0, t.shape[0]):
        temp = 0
        for group in categorical_results.keys():
            equal_opportunity_data[group] = utils.apply_threshold(categorical_results[group], t[i][temp])
            temp = temp + 1
            final_tprvalues[i].append(utils.get_true_positive_rate(equal_opportunity_data[group]))
        if (max(final_tprvalues[i]) - min(final_tprvalues[i]) <= epsilon):
            final_Ep_tprvalues.append(final_tprvalues[i])
            final_Ep_thresholds.append(t[i])

    max_accuracy = 0
    max_profit = -999999999999999
    total_accuracy_at_maxProfit=0
    final_thresholds_cost=[]
    for i in range(0, len(final_Ep_thresholds)):
        temp = 0
        for group in categorical_results.keys():
            equal_opportunity_data[group] = utils.apply_threshold(categorical_results[group],
                                                                  final_Ep_thresholds[i][temp])
            temp = temp + 1

        total_accuracy = utils.get_total_accuracy(equal_opportunity_data)
        if (total_accuracy > max_accuracy):
            max_accuracy = total_accuracy
            final_thresholds = final_Ep_thresholds[i]
            total_cost_at_maxacc = utils.apply_financials(equal_opportunity_data)
        total_cost = utils.apply_financials(equal_opportunity_data)
        if (total_cost > max_profit) :
            max_profit = total_cost
            final_thresholds_cost = final_Ep_thresholds[i]
            total_accuracy_at_maxProfit = utils.get_total_accuracy(equal_opportunity_data)
            final_equal_opportunity_data = copy.deepcopy(equal_opportunity_data)

    # print("Equal Opportunity Maximum accuracy is ",max_accuracy, "at thresholds",final_thresholds, ". Cost is",total_cost_at_maxacc)
    # print("Equal Opportunity Maximum Profit is ",max_profit, "at thresholds",final_thresholds_cost, ". Accuracy is",total_accuracy_at_maxProfit)

    i = 0
    for group in categorical_results.keys():
        thresholds[group] = final_thresholds_cost[i]
        i = i + 1

    # print("")
    # print("Equal Opportunity Maximum Profit is :",max_profit)
    # print("At thresholds :",thresholds)
    # print("Respective Accuracy is :",total_accuracy_at_maxProfit)

    # Must complete this function!


    # print("")
    # for group in final_equal_opportunity_data.keys():
    #     fscore_temp=utils.calculate_Fscore(final_equal_opportunity_data[group])
    #     print("Fscore for", group, fscore_temp)
    # print("")
    return final_equal_opportunity_data, thresholds
    # return equal_opportunity_data, thresholds
    # return None, None
#######################################################################################################################

"""Determines which thresholds to use to achieve the maximum profit or maximum accuracy with the given data
"""

def enforce_maximum_profit(categorical_results):
    mp_data = {}
    thresholds = {}
    cost= [[] for x in range(len(categorical_results.keys()))]

    max_cost=[-999999999999, -999999999999, -99999999999, -99999999999]
    thresholds_values=[0, 0, 0, 0]
    for i in np.arange(0, 1, 0.01): #put 0.001
        j=0
        for group in categorical_results.keys():
            mp_data[group]=utils.apply_threshold(categorical_results[group], i)
            cost[j].append(utils.apply_financials(mp_data[group], True))
            j=j+1
        for k in range (0,4) :
            if max_cost[k] < max(cost[k]):
                max_cost[k]= max(cost[k])
                thresholds_values[k]=i

    i=0
    for group in categorical_results.keys():
        thresholds[group]= thresholds_values[i]
        i=i+1

    temp=0
    for group in categorical_results.keys():
        mp_data[group] = utils.apply_threshold(categorical_results[group], thresholds_values[temp])
        temp = temp + 1
    # Must complete this function!
    #return mp_data, thresholds
    print("")
    for group in thresholds.keys():
        print("Threshold for " + group + ": " + str(thresholds[group]))

    print("")
    total_cost = utils.apply_financials(mp_data)
    total_accuracy = utils.get_total_accuracy(mp_data)

    # print("Maximum profit function results :")
    # print("Maximum Profit is :",total_cost)
    # print("At thresholds :",thresholds)
    # print("Respective Accuracy is :",total_accuracy)
    # print("")
    return mp_data, thresholds
    # return None,None
#######################################################################################################################
""" Determine thresholds such that all groups have the same PPV, and return the best solution
    according to chosen secondary optimization criteria
"""

def enforce_predictive_parity(categorical_results, epsilon):
    predictive_parity_data = {}
    initial_ppv = {}

    threshold = np.arange(0,1,0.01)
    for group in categorical_results.keys():
        initial_ppv[group] = []

    for i in range(0,len(threshold)):
        for group in categorical_results.keys():
            classification = categorical_results[group]
            predictive_parity_data[group] = utils.apply_threshold(classification, threshold[i])
            initial_ppv[group].append((utils.get_positive_predictive_value(predictive_parity_data[group]), threshold[i]))

    final_threshold = {}
    threshold_values = []
    ppv_12 = []
    ppv_23 = []

    for group in categorical_results.keys():
        final_threshold[group] = 0.0
        threshold_values = []

    for i in range(0,len(initial_ppv[list(initial_ppv)[0]])):
        for j in range(0,len(initial_ppv[list(initial_ppv)[1]])):
            if (((abs(initial_ppv[list(initial_ppv)[0]][i][0] - initial_ppv[list(initial_ppv)[1]][j][
                0]) <= epsilon)) and ((initial_ppv[list(initial_ppv)[0]][i][0],
                                       initial_ppv[list(initial_ppv)[0]][i][1],
                                       initial_ppv[list(initial_ppv)[1]][j][0],
                                       initial_ppv[list(initial_ppv)[1]][j][1]) not in ppv_12)):
                ppv_12.append((initial_ppv[list(initial_ppv)[0]][i][0], initial_ppv[list(initial_ppv)[0]][i][1],
                               initial_ppv[list(initial_ppv)[1]][j][0], initial_ppv[list(initial_ppv)[1]][j][1]))
                break

    for i in range(0,len(ppv_12)):
        for j in range(0,len(initial_ppv[list(initial_ppv)[2]])):
            if (utils.compare_probs((ppv_12[i][0], ppv_12[i][1]), initial_ppv[list(initial_ppv)[2]][j], epsilon)
                    and utils.compare_probs((ppv_12[i][2], ppv_12[i][3]), initial_ppv[list(initial_ppv)[2]][j],
                                      epsilon) == True
                    and ((ppv_12[i][0], ppv_12[i][1], ppv_12[i][2], ppv_12[i][3],
                          initial_ppv[list(initial_ppv)[2]][j][0],
                          initial_ppv[list(initial_ppv)[2]][j][1]) not in ppv_23)):
                ppv_23.append((ppv_12[i][0], ppv_12[i][1], ppv_12[i][2], ppv_12[i][3],
                               initial_ppv[list(initial_ppv)[2]][j][0], initial_ppv[list(initial_ppv)[2]][j][1]))
                break

    for i in range(0,len(ppv_23)):
        for j in range(0,len(initial_ppv[list(initial_ppv)[3]])):
            if (utils.compare_probs((ppv_23[i][0], ppv_23[i][1]), initial_ppv[list(initial_ppv)[3]][j], epsilon)
                    and utils.compare_probs((ppv_23[i][2], ppv_23[i][3]), initial_ppv[list(initial_ppv)[3]][j],
                                      epsilon) == True
                    and utils.compare_probs((ppv_23[i][4], ppv_23[i][5]), initial_ppv[list(initial_ppv)[3]][j],
                                      epsilon) == True
                    and ((ppv_23[i][1], ppv_23[i][3], ppv_23[i][5],
                          initial_ppv[list(initial_ppv)[3]][j][1]) not in threshold_values)):
                threshold_values.append(
                    (ppv_23[i][1], ppv_23[i][3], ppv_23[i][5], initial_ppv[list(initial_ppv)[3]][j][1]))
                break

    cost = 0
    max_cost = -999999999
    final_thresh = (0, 0, 0, 0)
    for i in range(len(threshold_values)):
        for group, j in zip(categorical_results.keys(), range(len(categorical_results.keys()))):
            predictive_parity_data[group] = utils.apply_threshold(categorical_results[group], threshold_values[i][j])
            # pass
        cost = utils.apply_financials(predictive_parity_data)
        if cost > max_cost:
            max_cost = cost
            final_thresh = threshold_values[i]

    for group, i in zip(categorical_results.keys(), range(len(categorical_results.keys()))):
        final_threshold[group] = final_thresh[i]

    for group in categorical_results.keys():
        predictive_parity_data[group] = utils.apply_threshold(categorical_results[group], final_threshold[group])

    return predictive_parity_data, final_threshold
""" Apply a single threshold to all groups, and return the best solution according to 
    chosen secondary optimization criteria
"""

def enforce_single_threshold(categorical_results):

    single_threshold_data = {}
    thresholds = {}
    max_cost=-999999999999
    for i in np.arange(0, 1, 0.01): # 0.01
        for group in categorical_results.keys():
            single_threshold_data[group]=utils.apply_threshold(categorical_results[group], i)
        total_cost = utils.apply_financials(single_threshold_data)
        if max_cost < total_cost:
            max_cost = total_cost
            x=i
    for group in categorical_results.keys():
        single_threshold_data[group] = utils.apply_threshold(categorical_results[group], x)
    total_accuracy = utils.get_total_accuracy(single_threshold_data)
    # print("Maximum Accuracy" , max_acc, "at threshold" , x , "with total cost", total_cost)
    i = 0
    for group in categorical_results.keys():
        thresholds[group] = x
        i = i + 1

    # print("")
    # print("Single threshold Maximum Profit is :",max_cost)
    # print("At thresholds :",thresholds)
    # print("Respective Accuracy is :",total_accuracy)

    return single_threshold_data, thresholds
    # return None,None

