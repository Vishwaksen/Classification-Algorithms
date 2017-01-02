# K-Nearest Neighbor Classification Algorithm 

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
        

def count_votes(nearest_neighbors):
    length = len(nearest_neighbors)
    predicted_label = {}
    for i in range(length):
        label = nearest_neighbors[i][-1]
        if label in predicted_label:
            predicted_label[label] = predicted_label[label] + 1
        else:
            predicted_label[label] = 1
    final_predicted_label = sorted(predicted_label.items(), key=operator.itemgetter(1), reverse=True)
    return final_predicted_label[0][0]

def calculate_euclidean_distance(train_data, test_data, length):
    euclidean_distance = 0
    counter = 0
    for (trn, tst) in itertools.zip_longest(train_data, test_data): 
        if(counter < length):
            try:
                x = float(trn)
                y = float(tst)
                euclidean_distance = euclidean_distance + math.pow(trn - tst, 2)
            except ValueError:
                if(trn == tst):
                    euclidean_distance = euclidean_distance + 0
                else:
                    euclidean_distance = euclidean_distance + 1
        counter = counter + 1
    dist = math.sqrt(euclidean_distance)
    return dist
    
    
def find_nearest_neighbors(train_data, test_data, k):
    distance = []
    for pos, val in enumerate(train_data):
        length = len(val)-1
        dist = calculate_euclidean_distance(val,test_data, length)
        distance.append((val,dist))
    distance.sort(key=operator.itemgetter(1))
    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(distance[i][0])
    return nearest_neighbors
    
file_name = "project3_dataset1.txt"
#file_name = "project3_dataset2.txt"

k = 10
k_fold_validation = 10

# Load the corresponding dataset to fetch records.
accuracy = 0
precision = 0
recall = 0
f_measure = 0
records = load_dataset(file_name)
for i in range(k_fold_validation):
    print("\nIteration " + str(i+1))
    train_data, test_data = form_training_testing_data(records,k_fold_validation, i+1)
    predicted_labels = []
    for pos, val in enumerate(test_data):
        nearest_neighbors = find_nearest_neighbors(train_data, val, k)
        label = count_votes(nearest_neighbors)
        predicted_labels.append(label)
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
print("\n\n***** KNN Classification Statistics Obtained Using " + str(k_fold_validation)+ "-Fold Cross Validation ***** ")
print("Accuracy : " + str(accuracy))
print("Precision : " + str(precision))
print("Recall : " + str(recall))
print("F-1 Measure : " + str(f_measure))
