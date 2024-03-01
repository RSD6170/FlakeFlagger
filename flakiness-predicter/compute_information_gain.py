from parallel_pandas import ParallelPandas
import pandas as pd
import numpy as np
import warnings
import sys
from sklearn.feature_extraction.text import CountVectorizer
import time


#%%
def main_inspection(data):    
    
    if ('flaky_source' in data.columns):
        data = data.drop(['flaky_source'], axis=1)
    data = data.dropna()
    lst_project = get_specific_project_data(data,1)
    new_data = data[data['project'].isin(lst_project)]
    new_data = new_data.drop(['Unnamed: 0', 'testClassName', 'testMethodName'], axis=1)
    return new_data

def get_specific_project_data (data, k):
    list_of_projects = []
    for i in data.project.unique():
        temp = data.loc[data['project'] == i]
        if ((temp['flaky'].values == 1).sum() >= k):
            list_of_projects.append(i)    
    return list_of_projects

#%%

def entropy_calculation(target_col):
    #Calculate the entropy of a dataset.
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name):
    
    #Calculate the entropy of target value .. 
    total_entropy = entropy_calculation(data[target_name])
        
    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy_calculation(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])

    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    
    return Information_Gain.round(5)
#%%

def tokenHelper(col, matrix_with_FlakyStatus):
    if str(col.name) != 'flaky':
        IG = InfoGain(matrix_with_FlakyStatus, col.name, 'flaky')
        return [col.name, "token", IG]

def vexctorizeTokenWithInformationGian(data,flakyStatus,IG_result_columns):
    
    vocabualry_vectorizer = CountVectorizer()
    train = vocabualry_vectorizer.fit_transform(data)
    matrix_token = pd.DataFrame(train.toarray(), columns = vocabualry_vectorizer.get_feature_names_out()).astype('int8')
    matrix_with_FlakyStatus = pd.concat([matrix_token, flakyStatus.reindex(matrix_token.index)], axis=1)

    # reduce the number of token in IG computation ..     
    matrix_with_FlakyStatus.drop([col for col, val in matrix_with_FlakyStatus[matrix_with_FlakyStatus.columns.difference(['flaky'])].sum().items() if val < 2], axis=1, inplace=True)
    
    print ("--- > start calculating the information gain for each token ")
    results = matrix_with_FlakyStatus.p_apply(tokenHelper, axis=0, args=(matrix_with_FlakyStatus,))

    print (":: Done from calculating the IG per token          ")

    results = results.T
    results.columns = IG_result_columns
    results.drop('flaky', axis=0, inplace=True)
    return results
    
    
#%%

def otherHelper(col, matrix_with_FlakyStatus, unwantedColumns, processedData):
    if col.name not in unwantedColumns:
        IG = InfoGain(matrix_with_FlakyStatus, col.name, 'flaky')
        if col.name not in processedData: return [col.name, "JavaKeyWords", IG]
        else: return [col.name, "FlakeFlagger", IG]

def calculateOtherFeaturesIG(data,unwantedColumns,processedData,IG_result_columns):
    print ("--- > start calculating the information gain for FlakeFlagger and Java keywords features  ")
    results = data.apply(otherHelper, axis=0, args=(data, unwantedColumns, processedData))
    print (":: Done from calculating the IG for Flakeflagger and Java keywords features  ")
    results = pd.DataFrame(np.vstack(results.drop(unwantedColumns)))
    results.columns = IG_result_columns
    results.drop('flaky', axis=0, inplace=True, errors='ignore')
    return results
    
#%%
def calculateOnlyFlakeFlaggerIG(data,unwantedColumns,IG_result):
    new_data = data.copy()
    counter = 0
    print ("--- > Start calculating the information gain for FlakeFlagger features  ")
    for col in new_data:
        if col not in unwantedColumns:
            IG = InfoGain(new_data,col,'flaky')
            counter = counter + 1
            total_counter = round((counter/len(new_data.columns))*100)
            sys.stdout.write('\r')
            sys.stdout.write(f" --> {int(total_counter)}{'% of the features have been processed ... '}")
            sys.stdout.write('\r')

            IG_result = IG_result.append(pd.Series([col,"FlakeFlagger",IG], index=IG_result.columns ), ignore_index=True)
    print (":: Done from calculating the IG for the Flakeflagger features  ")
    return IG_result
   
    

#%%
execution_time = time.time()

if __name__ == '__main__':

    ParallelPandas.initialize(n_cpu=12)
    vocabualry_and_processed_data = pd.read_csv("/home/ubuntu/atsfp/results_final/_generated_new/processed_data_with_vocabulary_per_test.csv")
    output = "/home/ubuntu/atsfp/results_final/_generated_new/Information_gain_per_feature.csv"
    #vocabualry_and_processed_data = pd.read_csv("/home/ubuntu/atsfp/FlakeFlagger/flakiness-predicter/result/processed_data_with_vocabulary_per_test.csv")
    #output = "test.csv"
    FlakeFlagger_features = ['assertion-roulette', 'conditional-test-logic', 'eager-test', 'fire-and-forget', 'indirect-testing', 'mystery-guest', 'resource-optimism', 'test-run-war', 'testLength', 'numAsserts', 'numCoveredLines', 'ExecutionTime', 'projectSourceLinesCovered', 'projectSourceClassesCovered', 'hIndexModificationsPerCoveredLine_window5', 'hIndexModificationsPerCoveredLine_window10', 'hIndexModificationsPerCoveredLine_window25', 'hIndexModificationsPerCoveredLine_window50', 'hIndexModificationsPerCoveredLine_window75', 'hIndexModificationsPerCoveredLine_window100', 'hIndexModificationsPerCoveredLine_window500', 'hIndexModificationsPerCoveredLine_window10000', 'num_third_party_libs']
    unwantedColumns = ['test_name', 'flaky', 'tokenList', 'Java_keywords', 'java_keywords', 'javaKeysCounter']

    IG_result_columns = ['features','type','IG']

    
    #IG for tokens only ...     
    IG_result_token = vexctorizeTokenWithInformationGian(vocabualry_and_processed_data['tokenList'],vocabualry_and_processed_data['flaky'],IG_result_columns)
    
    # IG for javakeywords and FlakeFlagger features...
    IG_result_java = calculateOtherFeaturesIG(vocabualry_and_processed_data,unwantedColumns,FlakeFlagger_features,IG_result_columns)

    IG_result = pd.concat([IG_result_token, IG_result_java], ignore_index=True)
    IG_result.to_csv(output,  index=False)
    
    print("The processed is completed in : (%s) seconds. " % round((time.time() - execution_time), 5))
