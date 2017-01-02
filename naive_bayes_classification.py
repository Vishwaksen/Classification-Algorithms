# Naive Bayes Classification Algorithm 

import numpy as np
import math as math
import random
from numpy import linalg as la
import operator
import itertools

# Loads the corresponding file into the association matrix
def load_dataset(file_name):
    records = []
    with open(file_name,'r') as f:
        for vals in f:
            splitter = vals.split()
            temp = []
            for attributes in splitter:
                try:
                    x = float(attributes)
                    temp.append(x)
                except ValueError:
                    temp.append(attributes)
            records.append(temp)
    print("Total Number of Records in " + str(file_name) + " Dataset : " + str(len(records)))
    return records

    
def form_training_testing_data(records, k_fold, ind):
    l = len(records)
    index = l / k_fold
    test_data = []
    train_data = []
    if(ind != 1 and ind != k_fold):
        start = int (index * (ind-1))
        end = int (index * ind)
    elif(ind == 1):
        start = 0
        end = int (index * ind)
    elif(ind == k_fold):
        start = int (index * (ind-1))
        end = l-1
    for i, j in enumerate(records):
        if(ind != 1 and ind != k_fold):
            if(i < start or i >= end):
                train_data.append(j)
            else:
                test_data.append(j)
        elif(ind == 1):
            if(i >= start and i < end):
                test_data.append(j)
            else:
                train_data.append(j)
        elif(ind == k_fold):
            if(i >= start and i <= end):
                test_data.append(j)
            else:
                train_data.append(j)
    print("Number of Records in Training Dataset : " + str(len(train_data)))
    print("Number of Records in Testing Dataset : " + str(len(test_data)))
    return train_data, test_data

def calculate_class_prior(train_data):
    length = len(train_data)
    class_prior = {}
    for i in range(length):
        label = train_data[i][-1]
        if label in class_prior:
            class_prior[label] = class_prior[label] + 1
        else:
            class_prior[label] = 1
    for key in class_prior.keys():
        val = class_prior.get(key)
        class_prior[key] = val / float(length)
        print("Class-" + str(key) + " Prior Probablity : " + str(class_prior[key]))
    return class_prior

def calculate_descriptor_posterior(test_data, train_data, label):
    denominator = 0 
    descriptor_posterior = 1
    hmap = {}       
    for pos, val in enumerate(train_data):
        if(val[-1] == label):
            denominator = denominator + 1
    hmap = map_descriptor_posterior_probablities(hmap, test_data, train_data, label)
    for key in hmap.keys():
        val = hmap.get(key)
        descriptor_posterior = (descriptor_posterior / float(denominator)) * val
    return descriptor_posterior
        

def map_descriptor_posterior_probablities(hmap, test_data, train_data, label):
    counter = 0
    length = len(test_data)
    for pos, val in enumerate(test_data):
        if(pos < length - 1):
            for p, q in enumerate(train_data):
                for i in range(len(q)):
                    if(i < length - 1):
                        if((i == pos) and (q[i] == val) and (q[-1] == label)):
                            if i in hmap:
                                hmap[i] = hmap[i] + 1
                            else:
                                hmap[i] = 1
    return hmap                                    

def calculate_mean_variance(train_data, position):
    hmap = {}
    """for pos, val in enumerate(train_data):
        l = len(val)
        for i, j in enumerate(val):
            if(i < l - 1):
                if(val[-1] == label):
                    if i in hmap:
                        y = hmap.get(i)
                        y.append(j)
                    else:
                        x = []
                        x.append(j)
                        hmap[i] = x """

    for pos, val in enumerate(train_data):
        label = val[-1]
        attribute = val[position]
        if label in hmap:
            tmp = hmap.get(label)
            tmp.append(attribute)
            hmap[label] = tmp
        else:
            temp = []
            temp.append(attribute)
            hmap[label] = temp
            
    statistics = {}
    for key in hmap.keys():
        x = hmap.get(key)
        mean = np.mean(x)
        variance = np.var(x)
        temp = []
        temp.append(mean)
        temp.append(variance)
        statistics[key] = temp
    return statistics

def calc_desc_post_prob(test_data, train_data, label):
    l = len(test_data)
    posterior = 1
    result = 1
    for pos, val in enumerate(test_data):
        if(pos < l - 1):
            if(isinstance(val,float)):
                #print("P : " + str(val))
                posterior = calculate_float_posterior(test_data, train_data, label, pos)
            elif(isinstance(val,str)):
                #print("Q : " + str(val))
                posterior = calculate_string_posterior(test_data, train_data, label, pos)
            result = result * posterior
    return result

def calculate_float_posterior(test_data, train_data, label, position):
    statistics = calculate_mean_variance(train_data, position)
    hmap = {}
    val = test_data[position]
    #list = statistics[position]
    #mean = list[0]
    #variance = list[1]
    stat_list = statistics.get(label)
    mean = stat_list[0]
    variance = stat_list[1]
    temp = 1.0 / (math.sqrt((2 * math.pi * variance)))
    numerator = (-(val - mean)**2) / (2 * variance)
    exp = math.exp(numerator)
    result = temp * exp
    return result

def calculate_string_posterior(test_data, train_data, label, position):
    statistics = calculate_string_statistics(train_data, position)
    value = statistics.get(label)
    length = len(value)
    attribute = test_data[position]
    count = 0
    for pos, val in enumerate(value):
        if(attribute == val):
            count = count + 1
    result = count / float(length)
    return result

def calculate_string_statistics(train_data, position):
    statistics = {}
    for pos, val in enumerate(train_data):
        label = val[-1]
        attribute = val[position]
        if label in statistics:
            tmp = statistics.get(label)
            tmp.append(attribute)
            statistics[label] = tmp
        else:
            temp = []
            temp.append(attribute)
            statistics[label] = temp
    return statistics

def calculate_descriptor_prior_probability(test_data, train_data):
    hmap = {}
    descriptor_prior = 1
    length = len(train_data)
    l = len(test_data)
    for pos, val in enumerate(test_data):
        if(pos < l - 1):
            for p, q in enumerate(train_data):
                for i in range(len(q)):
                    if(i < l - 1):
                        if((i == pos) and (q[i] == val)):
                            if i in hmap:
                                hmap[i] = hmap[i] + 1
                            else:
                                hmap[i] = 1
    for key in hmap.keys():
        val = hmap.get(key)
        descriptor_prior = (descriptor_prior / float(length)) * val
    return descriptor_prior

def calculate_statistics(test_data, predicted_labels):
    length = len(test_data)
    tn = 0
    tp = 0
    fn = 0
    fp = 0
    for i in range(length):
        if(predicted_labels[i] == test_data[i][-1]):
            if(predicted_labels[i] == 0):
                tn = tn + 1
            else:
                tp = tp + 1
        else:
            if(predicted_labels[i] == 0 and test_data[i][-1] == 1):
                fn = fn + 1
            elif(predicted_labels[i] == 1 and test_data[i][-1] == 0):
                fp = fp + 1
    try:
        accuracy = 100.0 * ((tp + tn) / float(tn + tp + fn + fp))
    except ZeroDivisionError:
        accuracy = 100.0 * (tp + tn)
    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = tp
    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = tp
    try:
        f_measure = (2 * tp) / float((2 * tp) + fn + fp)
    except ZeroDivisionError:
        f_measure = (2 * tp)
    return accuracy, precision, recall, f_measure

    
file_name = "project3_dataset1.txt"
#file_name = "project3_dataset2.txt"

k_fold_validation = 10

# Load the corresponding dataset to fetch records.
accuracy = 0
precision = 0
recall = 0
f_measure = 0
predicted_labels = []
records = load_dataset(file_name)
for i in range(k_fold_validation):
    print("\nIteration " + str(i+1))
    train_data, test_data = form_training_testing_data(records,k_fold_validation, i+1)
    predicted_labels = []
    class_prior_probablity = calculate_class_prior(train_data)
    
    for pos, val in enumerate(test_data):
        class_zero_prob = calc_desc_post_prob(val, train_data, 0)
        class_one_prob = calc_desc_post_prob(val, train_data, 1)
        
        descriptor_prior = calculate_descriptor_prior_probability(val, train_data)
        
        zero = class_zero_prob * float (class_prior_probablity.get(0))
        one = class_one_prob * float (class_prior_probablity.get(1))
        
        class_zero_final_prob = zero / float (descriptor_prior)
        class_one_final_prob = one / float (descriptor_prior)
        
        if(class_zero_final_prob > class_one_final_prob):
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)
    acc, prec, rec, f = calculate_statistics(test_data, predicted_labels)
    accuracy = accuracy + acc
    precision = precision + prec
    recall = recall + rec
    f_measure = f_measure + f
    print("*** Iteration " + str(i+1) + " Statistics *** ")
    print("Accuracy : " + str(acc))
    print("Precision : " + str(prec))
    print("Recall : " + str(rec))
    print("F-1 Measure : " + str(f))

accuracy = accuracy / float (k_fold_validation)
precision = precision / float (k_fold_validation)
recall = recall / float (k_fold_validation)
f_measure = f_measure / float (k_fold_validation)
print("\n\n***** Naive Bayes Classification Statistics Obtained Using " + str(k_fold_validation)+ "-Fold Cross Validation ***** ")
print("Accuracy : " + str(accuracy))
print("Precision : " + str(precision))
print("Recall : " + str(recall))
print("F-1 Measure : " + str(f_measure))
