
import time
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, f_regression
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


    

def create_doc():
    
    excel = pd.read_csv(r"C:\Users\tomar\Documents\Project\git\dataMineWithGap.csv")
        
    excel['last_run'] =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    
    excel['last_result'] = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    
    # print(excel['last_result'])
    # feature_columns["LastResults"] = last_results
    
    excel['times_ran'] = [int(len(x)/3) for x in excel['LastResults'].to_list()]
    
    excel['gap_cats'] = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), excel['GapInRun'])))
    
    del excel['Unnamed: 22']
    del excel['Unnamed: 23']
    del excel['LastResults']
    del excel['LastRun']
    del excel["Unnamed: 0"]
    del excel["Id"]
    # excel.drop('Unnamed: 0', axis=1) #
    
    
    excel['my_maturity'] = calc_maturity(excel)
    

    # Initialize MinMaxScaler
    # Fit and transform the column you want to normalize
    # Let's normalize column 'B'
    
    # for column in excel:
    #     print(column)
    
    excel.to_csv(r'classification\ranking\my_data_mutual_info3P3.csv', index=False)
    

def calc_maturity(excel):
    global calculation
    
    # df = calculate_all_mutuals(excel)

    MOST_TIMES_RAN = excel['times_ran'].max()
    
    LEAST_TIMES_RAN = 4
    
    """
    TO FIND MOST AND LEAST
    last_results = excel["LastResults"].to_numpy()
    
    leng = [len(x) for x in last_results]
    
    print(sorted(leng, reverse=True)[0])"""
    
    # many times ran
    # ran at the end
    # high failure precentage
    # compare the max and min num of ran
    month = excel["Month"]
    fp = excel["Failure_Percentage"]
    
    # last_results = excel["lastResults"]
    times_ran = excel["times_ran"]
    
    in_same_cycle = excel["InSameCycle"]
    cycle = excel["Cycle"]
    gap_in_run = excel["GapInRun"]
    
    duration = excel["Duration"]
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap_in_run)))
    
    matu1 =(month * cycle * (in_same_cycle + 1))/ (gap_cats + 1) + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))
    
    # matu1 = cycle + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))

    
    
    
    verdict = excel["Verdict"]
    # feature_names, mutual_info_list = mat(df1,verdict)
    
    calculation = "(month * cycle * (in_same_cycle + 1))/ (gap_cats + 1) + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))"
    
    # calc_mutual_info(excel.columns.tolist(), verdict)
    
    
    
    # st = ""
    # for col in range(len(mutual_info_list)):
    #     st += f"{feature_names[col]}:{mutual_info_list[col]}\n"
    
    # with open("classification\maturity_trials.txt", "a") as f:
    #     f.write(f"calculation = {calculation}\n{st}\n\n\n")
        
        
    return matu1   


def calc_mutual_info(X_train, y_train):
    mutual_info = mutual_info_regression(X_train, y_train)
    
    k = 30  # Set the number of top features you want to select

    best_features = np.argsort(mutual_info)[::-1][:k]

    selected_features = X_train.columns[best_features]

    X_mi_selected = X_train[selected_features]
    
    print("\nSelected features based on mutual information:")

    print(X_mi_selected.columns)
    
    plt.figure(figsize=(10, 6))

    plt.barh(X_mi_selected.columns, mutual_info[best_features])

    st = ""
    for col in range(len(X_mi_selected.columns)):
        st += f"{X_mi_selected.columns[col]}:{mutual_info[best_features][col]}\n"
        
    # with open(r'classification\ranking\mutual_infos\mutual_infos_verdict.txt', 'a') as f:
    #     # f.write(calculation + "\n")
    #     f.write(st + "\n\n")
        
        
    return X_mi_selected.columns, mutual_info[best_features]
        
def graph(feature_names, mutual_info_list):
               
    plt.xlabel('Mutual Information Score')

    plt.ylabel('Feature Names')

    plt.title('Mutual Information for Feature Selection')

    plt.show()
    
    # Bar Chart
    
    print("Bar Chart ____")
    
    # Create the figure and axis

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create the horizontal bar chart

    bars = ax.barh(feature_names, mutual_info_list)
    
    # Set the labels and title

    ax.set_xlabel('Mutual Information Score')

    ax.set_ylabel('Feature Names')

    ax.set_title('Mutual Information for Feature Selection')



    # Annotate scores on each bar

    for bar, score in zip(bars, mutual_info_list):

        width = bar.get_width()

        label_x_pos = width + 0.01  # Adjust the position for the score

        ax.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{score:.6f}', ha='center', va='center')
    
    plt.show()
    



# # create_doc()
# excel = pd.read_csv('classification/ranking/my_data_mutual_info3P3.csv')

# normalise = ['Num_Previous_Execution','GapInRun','CycleRun','last_run','times_ran','gap_cats','my_maturity', 'Cycle']
# logit = ['last_run', 'GapInRun']

# scaler = MinMaxScaler()
# for column in excel.columns:
#     if column in normalise:  # Check if the column contains numerical data
#         excel[column + '_normalized'] = scaler.fit_transform(excel[[column]])
#         excel[column + "_normalized"] = excel[column + "_normalized"] * 10
#         del excel[column]
        
#     if column in logit:
#         offset = 1e-9
#         excel[column + "_normalised_log"] = np.log(excel[column + "_normalized"] + offset)
#         del excel[column + '_normalized']
        
        


# excel = pd.read_csv('classification/ranking/my_data_mutual_info3P3.csv')

# # excel = pd.read_csv('normalised_2_log.csv')

# # excel.to_csv('normalised_4_log.csv', index=False)

# verdict = excel['Verdict']
# calcPrio = excel['CalcPrio']
# rank = excel['Rank']




# original = ["Duration","CalcPrio","Verdict","Cycle","Month","Quarter","Failure_Percentage","Rank"]

# without_target_and_redundent = ["Name","Duration","Cycle","Month","Quarter","Failure_Percentage","GapInRun","InSameCycle","CycleRun","last_run","last_result","times_ran","gap_cats","my_maturity"]
# without_target_and_redundent_and_maturity = ["Name","Duration","Cycle","Month","Quarter","Failure_Percentage","GapInRun","InSameCycle","CycleRun","last_run","last_result","times_ran","gap_cats"]

# columns = ['Name_normalized','Duration_normalized','CalcPrio_normalized','Verdict_normalized','Cycle_normalized','DurationGroup_normalized','my_maturity_normalized','TimeGroup_normalized','Month_normalized','Quarter_normalized','Failure_Percentage_normalized','Num_Previous_Execution_normalized','Maturity_Level_normalized','Rank_score_normalized','Original_Index_normalized','Rank_normalized','GapInRun_normalized','InSameCycle_normalized','CycleRun_normalized','last_run_normalized','last_result_normalized','times_ran_normalized','gap_cats_normalized']

# # prioCols = ['Duration','Cycle','my_maturity','Month','Failure_Percentage','last_run','times_ran','GapInRun']

# verdictCols = ['Cycle_normalized','my_maturity_normalized','Month','Failure_Percentage','GapInRun_normalised_log','last_run_normalised_log','last_result','times_ran_normalized']

# rankingCols = ['Duration','Cycle_normalized','my_maturity_normalized','Month','Failure_Percentage','last_run_normalised_log','times_ran_normalized', 'Num_Previous_Execution_normalized']

# excel = excel[without_target_and_redundent_and_maturity]

# feature_names, info_list = calc_mutual_info(excel, calcPrio)
# graph(feature_names, info_list)

# feature_names, info_list = calc_mutual_info(excel, verdict)
# graph(feature_names, info_list)

# feature_names, info_list = calc_mutual_info(excel, rank)
# graph(feature_names, info_list)

# input()
    
