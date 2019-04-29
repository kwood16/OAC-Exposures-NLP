# import packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# input files
NLP_RESULTS = './results/combined_dataframe.csv'
FULL_DATASET = './results/labeled_training_set.csv'

#output files
OUT_FILE = './results/performance.txt'
file = open(OUT_FILE, 'w')

# load data
nlp_frame = pd.read_csv(NLP_RESULTS)
data_frame = pd.read_csv(FULL_DATASET)

# get performance results
def get_results(dataframe, goldstd, predicted):
    truepos = len(dataframe[(dataframe[goldstd] == 1) & (dataframe[predicted] == 1)])
    trueneg = len(dataframe[(dataframe[goldstd] == 0) & (dataframe[predicted] == 0)])
    falsepos = len(dataframe[(dataframe[goldstd] == 0) & (dataframe[predicted] == 1)])
    falseneg = len(dataframe[(dataframe[goldstd] == 1) & (dataframe[predicted] == 0)])

    sensitivity = truepos / (truepos + falseneg) * 100
    specificity = trueneg / (trueneg + falsepos) * 100
    ppv = truepos / (truepos + falsepos) * 100
    npv = trueneg / (trueneg + falseneg) * 100
    accuracy = (truepos + trueneg) / (truepos + trueneg + falsepos + falseneg) *100

    print ('Accuracy:\t\t\t' + str(accuracy))
    print ('Sensitivity:\t\t\t' + str(sensitivity))
    print ('Specificity:\t\t\t' + str(specificity))
    print ('Positive Predicted Value:\t' + str(ppv))
    print ('Negative Predicted Value:\t' + str(npv) + '\n')
    print ('Contingency Table: \n' + str(truepos) + '\t' + str(falsepos) + '\n' + str(falseneg) + '\t' + str(trueneg))
    file.write ('Accuracy:\t\t\t' + str(accuracy) + '\n')
    file.write ('Sensitivity:\t\t\t' + str(sensitivity) + '\n')
    file.write ('Specificity:\t\t\t' + str(specificity) + '\n')
    file.write ('Positive Predicted Value:\t' + str(ppv) + '\n')
    file.write ('Negative Predicted Value:\t' + str(npv) + '\n')
    file.write ('Contingency Table: \n' + str(truepos) + '\t' + str(falsepos) + '\n' + str(falseneg) + '\t' + str(trueneg))

# nlp
print ('NLP Performance: ')
file.write ('NLP Performance: \n')
get_results(nlp_frame, 'goldstd_class', 'predicted_class')
