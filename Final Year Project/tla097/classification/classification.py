import random
import pandas as pd
from sklearn import ensemble, svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import f1_score 
import numpy as np
import time
import datetime
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import VotingClassifier

from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

  
  
def normalise(df):
    columns_to_normalize = df.columns.to_list()
    columns_to_normalize.remove("Verdict")
    columns_to_normalize.remove("CalcPrio")
    columns_to_normalize.remove("Rank")
    min_max_scaler = MinMaxScaler()
    selected_columns = df[columns_to_normalize]
    min_max_scaler = MinMaxScaler()
    normalized_columns = min_max_scaler.fit_transform(selected_columns)

    # Replace the original columns with the normalized values
    df[columns_to_normalize] = normalized_columns
    
    for col in df.columns.tolist():
        if col == "GapInRun" or col == "last_run": 
            df[col] = np.log(df[col] + 1e-8)

    df.to_csv("all_normed_logged6.csv")
    

def svmo(X,y,t, X_train, X_test, y_train, y_test, search = False):
    parameters = {}
    if search:
        parameters = {
                      'kernel' : ['poly', 'rbf', 'sigmoid'], 
                      'C': [0.1, 1, 10], 
                      'gamma': ['scale', 'auto'], 
                      'degree' : [3,5,7]
                      }

    best = 0
    with open("svm_grid_search.txt", "a") as f:
        f.write(f"{t.columns}\n")
    pretime = time.time()
    clf = model_selection.GridSearchCV(svm.SVC(), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    posttime = time.time() - pretime
    print(f"time = {posttime}")
    with open("svm_grid_search.txt", "a") as f:
        f.write(f"score = {score}, params = {parameters}, time = {posttime}")

    print(f"score = {score}")
    if score > best:
        best = score
        with open("svm_grid_search.txt", "a") as f:
            f.write(f"                                BEST = {score}\n")
    else:
        with open("svm_grid_search.txt", "a") as f:
            f.write(f"\n")
            
    return clf.best_params_
    # print(X)
    
    
    
    # pretime = time.time()
    # params = {'kernel' : 'poly'}
    # score = model_selection.cross_val_score(svm.SVC(**params), X, y, cv=10)
    # posttime = time.time() - pretime
    # string = f"svm = {score}, time = {posttime}, params = {params}"
    # print(string)
    
    # with open("svm.txt", "a") as f:
    #     f.write(string)
    # # print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))
    
    
    # print('Accuracy train (Polynomial Kernel): ', "%.2f" % (poly_accuracy_tr*100))
###################################################################################
    # from sklearn.naive_bayes import GaussianNB 
    # gnb = GaussianNB().fit(X_train, y_train) 
    # gnb_predictions = gnb.predict(X_test) 
    

def bayes(X,y):
####################################################################################################################
    from sklearn.naive_bayes import GaussianNB 
    
    pretime = time.time()
    params = {}
    score = np.mean(model_selection.cross_val_score(GaussianNB(), X_train, y_train, cv=5))
    posttime = time.time() - pretime
    string = f"bayes = {score}, time = {posttime}, params = {params}"
    print(string)
    
    with open("bayes.txt", "a") as f:
        f.write(string)
        
    return 1
    
    
    
    # clf = svm.SVC(decision_function_shape='ovo')
    # clf.fit(X_train, y_train)
    # dec = clf.decision_function([X_test])
    # print(dec.shape[1]) # 4 classes: 4*3/2 = 6
    
    # clf.decision_function_shape = "ovr"
    # dec = clf.decision_function([[1]])
    # dec.shape[1] # 4 classes

    
    
    
    
    # rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
    # rbf_pred = rbf.predict(X_test)

    
    # rbf_accuracy = accuracy_score(y_test, rbf_pred)
    # rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    # print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
    # print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))
    
    
    
def calculate_all_mutuals(excel):

    feature_columns = {}
    column_names = ["Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution","Verdict"]    
    
    for column_name in column_names:
        column = excel[column_name].to_numpy()
        feature_columns[column_name] = column 
        
    
    
    last_results = [int(element[1]) for element in excel["LastResults"].to_numpy()]
    time_ran =  [time.mktime(datetime.datetime.strptime(element, "%d/%m/%Y %H:%M").timetuple()) for element in excel["LastRun"].to_numpy()]
    gap = excel["GapInRun"].to_numpy()
    # gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 2), gap)))
    
    ##########

    
    #########
    
    
    feature_columns["LastRan"] = time_ran
    feature_columns["LastResults"] = last_results
    feature_columns["gap"] = gap
    
    last_results_len = excel["LastResults"].to_numpy()
    times_ran = np.array([len(x) for x in last_results_len])
    
    MOST_TIMES_RAN = 1186
    LEAST_TIMES_RAN = 4
    
    #############
    month = excel["Month"].to_numpy()
    fp = excel["Failure_Percentage"].to_numpy()
    
    last_results = excel["LastResults"].to_numpy()
    times_ran = np.array([len(x) for x in last_results])
    
    duration  = excel["Duration"]
    
    in_same_cycle = excel["InSameCycle"]
    cycle = excel["Cycle"]
    cycle_run = excel["CycleRun"]
    gap_in_run = excel["GapInRun"]
    gap_cats = np.array(list(map(lambda x : 0 if x < 1 else (1 if x < 2 else 3), gap_in_run)))
    
    div = np.array(list(map(lambda c,g: min(c,g), in_same_cycle, gap_in_run)))
    
    matu1 = (month * cycle * (in_same_cycle + 1))/ (gap_cats + 1) + fp * ((times_ran - LEAST_TIMES_RAN)/(MOST_TIMES_RAN))
    #############
    
    feature_columns["My_Maturity"] = matu1
    feature_columns["Duration"] = excel["Duration"]
    
    df = pd.DataFrame.from_dict(feature_columns)
    
    return df



def anis(X, y):
    
    
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2)
    
    # Make it an OvR classifier
    OvR = OneVsRestClassifier(svm.SVC(kernel='rbf', gamma=0.001, C=1000))
    # Fit the data to the OvR classifier
    OvR = OvR.fit(X_train, y_train)
    OvR_predict = OvR.predict(X_test)
    print ("--------------OvR classifier Evaluation---------------------")
    # Evaluating the model
    print(f"Test Set Accuracy : {accuracy_score(y_test, OvR_predict) * 100} %\n\n")
    print(f"Classification Report : \n\n{classification_report(y_test, OvR_predict)}")

    # Make it an OvO classifier
    ovo_classifier = OneVsOneClassifier(svm.SVC(kernel='rbf', gamma=0.001, C=1000))
    #Fit the data to the OvO classifier
    ovo_classifier = ovo_classifier.fit(X_train, y_train)
    OvO_predict = ovo_classifier.predict(X_test)
    print ("--------------OvO classifier Evaluation---------------------")
    # Evaluating the model
    print(f"Test Set Accuracy : {accuracy_score(y_test, OvO_predict) * 100} %\n\n")
    print(f"Classification Report : \n\n{classification_report(y_test, OvO_predict)}")
    

def gbrm(X,y,t, X_train, X_test, y_train, y_test, search = False):
    N_ESTIMATORS = 100
    MAX_DEPTH = 3
    LEARNING_RATE = 1
    parameters = {}
    if search:
        parameters = {
                                "loss":['log_loss'],
                                "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                                "min_samples_split": np.linspace(0.1, 0.5, 12),
                                "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                                "max_depth":[3,5,8],
                                "max_features":["log2","sqrt"],
                                "criterion": ["friedman_mse", "squared_error"],
                                "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                                "n_estimators":[10]
    }

    best = 0
    with open("gbrm.txt", "a") as f:
        f.write(f"{t.columns}\n")
    pretime = time.time()
    clf = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score)
    posttime = time.time() - pretime
    print(f"time = {posttime}")
    with open("gbrm.txt", "a") as f:
        f.write(f"score = {score}, params = {parameters}, time = {posttime}")

    print(f"score = {score}")
    if score > best:
        best = score
        with open("gbrm.txt", "a") as f:
            f.write(f"                                BEST = {score}\n")
    else:
        with open("gbrm.txt", "a") as f:
            f.write(f"\n")
            
    return clf.best_params_
        
    # Grid Search
    # for n_est in np.arange(10, 31, 10):
    #     for m_d in np.arange(3, 12, 2):
    #         for lr in np.arange(0.01, 1.41,0.1):
    #             params = {'n_estimators' : n_est, 'max_depth': m_d, 'learning_rate' : lr}
    #             grad_b_c = ensemble.GradientBoostingClassifier(**params)
    #             score = cross_val_score(grad_b_c, X, y, scoring= 'accuracy', cv = 10)
                
                             
    #             if score > biggest[0]:
    #                 biggest = [score, n_est, m_d, lr]
    #                 print(f"Current Biggest = {score}, n_est = {n_est}, m_d = {m_d}, lr = {lr}\n")
    #                 with open("gradient_booster_classifier4.txt", "a") as f:
    #                     f.write(f"\n\nCurrent Biggest = {score}, n_est = {n_est}, m_d = {m_d}, lr = {lr}\n\n")
                
    #             print(f"n_estimators:{n_est}, max_depth:{m_d}, learning rate:{lr} -> score = {score}\n")
    #             with open("gradient_booster_classifier4.txt", "a") as f:
    #                 f.write(f"n_estimators:{n_est}, max_depth:{m_d}, learning rate:{lr} -> score = {score}\n")
    
            
    # # with open("gradient_booster_classifier2.txt", "a") as f:
    # #     f.write(f"n_estimators:{N_ESTIMATORS}, max_depth:{MAX_DEPTH}, learning rate:{LEARNING_RATE} -> score = {grad_b_r_m.score(X_test, y_test)}\n")
    
    # parameters = {
    # "loss":['log_loss'],
    # "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    # "min_samples_split": np.linspace(0.1, 0.5, 12),
    # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    # "max_depth":[3,5,8],
    # "max_features":["log2","sqrt"],
    # "criterion": ["friedman_mse", "squared_error"],
    # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    # "n_estimators":[10]
    # }
    # ran = 0
    # while ran < 3000:
    #     loss = random.choice(parameters["loss"])
    #     lr = random.choice(parameters["learning_rate"])
    #     mss = random.choice(parameters["min_samples_split"])
    #     msl = random.choice(parameters["min_samples_leaf"])
    #     md = random.choice(parameters["max_depth"])
    #     mf = random.choice(parameters["max_features"])
    #     c = random.choice(parameters["criterion"])
    #     s = random.choice(parameters["subsample"])
    #     nest = random.choice(parameters["n_estimators"])
    # params = {'loss' : loss, 'n_estimators' : nest, 'max_depth': md, 'learning_rate' : lr, 'min_samples_split' : mss, 'min_samples_leaf' : msl, 'max_features' : mf, 'criterion' : c, 'subsample' : s}

    # params = {'n_estimators' : 10, 'max_depth': 5, 'learning_rate' : 0.76}
    # grad_b_c = ensemble.GradientBoostingClassifier(**params)
    # score = np.mean(cross_val_score(grad_b_c, X, y, scoring= 'accuracy', cv = 10))
    
                    
    # if score > biggest[0]:
    #     biggest = (score, params)
    #     print(f"Current Biggest = {score}, {params}\n")
    #     with open("gradient_booster_classifier_new_test.txt", "a") as f:
    #         f.write(f"\n\nCurrent Biggest = {score}, {params}\n\n")
    
    # print(f"Accuracy: {score} [Gradient Boosting Classifier]")
    # print(f"{params} -> score = {score}\n")
    # with open("gradient_booster_classifier_new_test.txt", "a") as f:
    #     f.write(f"{params} -> score = {score}\n")
            
        # ran +=1

            
    # with open("gradient_booster_classifier2.txt", "a") as f:
    #     f.write(f"n_estimators:{N_ESTIMATORS}, max_depth:{MAX_DEPTH}, learning rate:{LEARNING_RATE} -> score = {grad_b_r_m.score(X_test, y_test)}\n")
    
def grid_search(trainX, trainY, testX,testY):
    parameters = {
    "loss":['log_loss'],
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse", "squared_error"],
    "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    "n_estimators":[10]
    }

    # clf = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

    # clf.fit(trainX, trainY)
    # # Get the best hyperparameters
    # best_params = clf.best_params_
    # print("Best hyperparameters:", best_params)
    # print(clf.score(testX, testY))
    # print(clf.best_params_)
    
    # Define the hyperparameters grid
    # Initialize GridSearchCV
    clf = model_selection.GridSearchCV(ensemble.GradientBoostingClassifier(), parameters, cv=10, n_jobs=-1)

    # Fit the grid search object to the data
    clf.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = clf.best_params_
    print("Best hyperparameters:", best_params)

    # Get the best cross-validation score
    best_score = clf.best_score_
    print("Best cross-validation score:", best_score)

    # Get the best model
    best_model = clf.best_estimator_
    print(best_model.score(X_test, y_test))

    # Use the best model to make predictions
    predictions = best_model.predict(X_test)

def voting(X,y, X_test, y_test, X_train, y_train, rand, svm_dict, bayes, gbrm):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier

    # params= {'n_estimators' : 10, 'max_depth': 5, 'learning_rate' : 0.76}
    # clf1 = ensemble.GradientBoostingClassifier(**params)
    # clf2 = RandomForestClassifier(n_estimators=1000, random_state=1)

    # eclf = VotingClassifier(estimators=[("gbc", clf1), ("rf", clf2)], voting='hard')

    # for clf, label in zip([clf1, clf2, eclf], ['Gradient Boosting Classfier', 'Random Forest Classifier', "Voting Classifier"]):
    #     scores = cross_val_score(clf, X, y, scoring='accuracy', cv=10)
    #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        
    # eclf.fit(X_train, y_train)
    # print(eclf.score(X_test, y_test))
    # scores = cross_val_score(eclf, X, y, scoring= 'accuracy', cv = 10)
    # print(f"Accuracy: {scores} [Cross Validation]")
    
    clf1 = ensemble.GradientBoostingClassifier(**gbrm)
    clf2 = RandomForestClassifier(**rand)
    clf3 = GaussianNB()
    clf4 = svm.SVC(**svm_dict)

    eclf = VotingClassifier(estimators=[("gbc", clf1), ("rf", clf2), ("g", clf3), ("svm", clf4)], voting='hard')

    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Gradient Boosting Classfier', 'Random Forest Classifier', "guassian", "svm" , "Voting Classifier"]):
        pretime  = time.time()
        scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
        postime = time.time() - pretime
        with open("voting.txt", "a") as f:
            f.write("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
            f.write(f" time = {postime}")
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        print(f" time = {postime}")
    
    pretime= time.time()
    eclf.fit(X_train, y_train)
    postime = time.time() - pretime
    print(eclf.score(X_test, y_test))
    scores = cross_val_score(eclf, X, y, scoring= 'accuracy', cv = 5)
    print(f"Accuracy: {scores} [Cross Validation]")
    with open("voting.txt", "a") as f:
        f.write(f"Accuracy: {scores} [Cross Validation]")
        f.write(f" time = {postime}")
    print(f"Accuracy: {scores} [Cross Validation]")
    print(f" time = {postime}")

def rand_forest(X,y,t, X_train, X_test, y_train, y_test, search= False):        
    parameters = {}
    if search:
        parameters = {'n_estimators' : list(range(10, 300, 10))}

    best = 0
    with open("forest_grid_search.txt", "a") as f:
        f.write(f"{t.columns}\n")
    pretime = time.time()
    clf = model_selection.GridSearchCV(ensemble.RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    posttime = time.time() - pretime
    print(f"time = {posttime}")
    with open("forest_grid_search.txt", "a") as f:
        f.write(f"score = {score}, params = {parameters}, time = {posttime}")

    print(f"score = {score}")
    if score > best:
        best = score
        with open("forest_grid_search.txt", "a") as f:
            f.write(f"                                BEST = {score}\n")
    else:
        with open("forest_grid_search.txt", "a") as f:
            f.write(f"\n")
            
    return clf.best_params_
                
    

            
    
            
        
    
def k_fold(X,y):
    splits = 10
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=splits)

    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
    print(X_train)
    print(X_test)
    
    for test in range(10):
        with open("gradient_booster_classifier_cross_val_results.txt", "a") as f:
            f.write(f"test_num = {test} - {gbrm(X_test[test], y_test[test], X_train[test], y_train[test])}\n")
    
        
        
    
    
    
    
# excel = pd.read_csv(r"C:\Users\tomar\Documents\Project\git\dataMineWithGap.csv")

# excel = pd.read_csv("RL/rlq/my_data_mutual_info3P3.csv")

# feature_columns = ["Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution","Verdict", "last_run", "last_result", "gap_cats", "my_maturity", "Duration"]
# Name,Duration,CalcPrio,Verdict,Cycle,DurationGroup,TimeGroup,Month,Quarter,Failure_Percentage,Num_Previous_Execution,Maturity_Level,Rank_score,Original_Index,Rank,GapInRun,InSameCycle,CycleRun,last_run,last_result,times_ran,gap_cats,my_maturity


# feature_columns = ["Cycle", "Month", "Failure_Percentage", "Num_Previous_Execution", "last_run", "last_result", "gap_cats", "my_maturity", "Duration"]

# # Name,Duration,CalcPrio,Verdict,DurationGroup,TimeGroup,Month,Quarter,Failure_Percentage,Maturity_Level,Rank_score,Original_Index,Rank,InSameCycle,last_result,Cycle_normalized,Num_Previous_Execution_normalized,GapInRun_normalised_log,CycleRun_normalized,last_run_normalised_log,times_ran_normalized,gap_cats_normalized,my_maturity_normalized

# prioColsNorm = ['Verdict','Cycle_normalized','my_maturity_normalized','Month','Failure_Percentage','last_run_normalised_log','times_ran_normalized']
# prioCols = ['Cycle','my_maturity','Month','Failure_Percentage','last_run','




def classify_function(X,y,X_train, X_test, y_train, y_test, model, params, name, filename):
        classifier = model_selection.GridSearchCV(model, params, cv=5, n_jobs=-1)
        pretime = time.time()
        classifier.fit(X_train, y_train)
        posttime = time.time() - pretime
        score = classifier.score(X_test, y_test)
        result_string = f"score = {score}, time = {posttime} [{name}] params = {classifier.best_params_}\n"
        print(result_string)
        with open (filename, "a") as f:
            f.write(result_string)
            
        return classifier.best_estimator_
        

    

def classify_all(X, y, name, default = True):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.8, test_size=0.2)
    gbrm_params =  {
                    "loss":['log_loss'],
                    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    # "min_samples_split": np.linspace(0.1, 0.5, 12),
                    # "min_samples_leaf": np.linspace(0.1, 0.5, 12),
                    # "max_depth":[3,5,8],
                    # "max_features":["log2","sqrt"],
                    # "criterion": ["friedman_mse", "squared_error"],
                    # "subsample":[0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
                    "n_estimators":list(range(10, 300, 20))
    }
    grad_b_c = ensemble.GradientBoostingClassifier()
    
    svm_model = svm.SVC()
    
    svm_params = {
        'C':  [0.1] + list(np.arange(0.2, 1.7, 0.2)),
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'decision_function_shape' :['ovo', 'ovr']
}
    
    
    rand_forest_model = ensemble.RandomForestClassifier()
    forset_params = {
    'n_estimators': list(range(10, 300, 10)), 
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],   
}
    
    gnb_model = GaussianNB()
    bayes_params = {}
    
    params  = [{},{},{},{}]
    if not default:
        params = [gbrm_params, svm_params, forset_params, bayes_params]
    models = [grad_b_c, svm_model, rand_forest_model, gnb_model]
    names = ["Gradient Boosted Classifier", "SVM Classifier", "Random Forest Classifer", "Naive Bayes Classifier"]
    
    estimators = []
    for i in range(len(names)):
        cla = classify_function(X, y, X_train, X_test, y_train, y_test, models[i], params[i], names[i], name)
        estimators.append(cla)
        
    
    voting_zip = list(zip(names, models))
    eclf = VotingClassifier(estimators=voting_zip, voting='hard')
    pretime = time.time()
    eclf.fit(X_train, y_train)
    score = eclf.score(X_test, y_test)
    posttime = time.time() - pretime
    result_string = f"Accuracy: {score}, time = {posttime}: [Cross Validation] children = {names}\n"
    print(result_string)
    with open (name, "a") as f:
            f.write(result_string)
    
    
    
    params = [gbrm_params, forset_params]
    models = [grad_b_c, rand_forest_model]
    names = ["Gradient Boosted Classifier", "Random Forest Classifer"]
    
    
    voting_zip = list(zip(names, models))
    eclf = VotingClassifier(estimators=voting_zip, voting='hard')
    pretime = time.time()
    eclf.fit(X_train, y_train)
    score = eclf.score(X_test, y_test)
    posttime = time.time() - pretime
    result_string = f"Accuracy: {score}, time = {posttime}: [Cross Validation] children = {names}\n"
    print(result_string)
    with open (name, "a") as f:
            f.write(result_string)
    
    
    

    
    
    
    
    

# anis(X, y)

# # 
# voting(X,y, X_test, y_test ,X_train, y_train)

# # grid_search(X_train, y_train, X_test, y_test)
# # # k_fold(X,y)

# voting_mod = voting(X,y, X_test, y_test ,X_train, y_train, rand, svm_mod, bayes_mod, gbrm_mod)

# # anis(X, y)

# svmo(X,y)
# gbrm(X,y)
# prioColsNorm = ['Verdict','Cycle_normalized','my_maturity_normalized','Month','Failure_Percentage','last_run_normalised_log','times_ran_normalized']
# excel = pd.read_csv("RL/rlq/my_data_mutual_info3P3.csv")
# prioCols = ['Cycle','my_maturity','Month','Failure_Percentage','last_run','times_ran']
# prioColsMore = ['Cycle','my_maturity','Month','Failure_Percentage','last_run','times_ran', 'Duration', 'GapInRun']
# # excel =pd.read_csv("normalised_4_log.csv")

# cols = excel[prioCols].columns.tolist()
# y = excel["CalcPrio"].to_numpy()
# X = excel[prioCols].to_numpy()
# with open ("proritisation1.txt", "a") as f:
#             f.write(str(cols) + "\n")

# classify_all(X, y)
# classify_all(X, y, default=False)

# y = excel["CalcPrio"].to_numpy()
# X = excel[prioColsMore].to_numpy()
# cols = excel[prioColsMore].columns.tolist()
# with open ("proritisation1.txt", "a") as f:
#             f.write(str(cols))


# classify_all(X, y)
# classify_all(X, y, default=False)


# excel = pd.read_csv("RL/rlq/my_data_mutual_info3P3.csv")
# normalise(excel)


# excel = pd.read_csv("normalised_4_log.csv")
prioCols = ['Cycle','my_maturity','Month','Failure_Percentage','last_run','times_ran']
prioColsMore = ['Cycle','my_maturity','Month','Failure_Percentage','last_run','times_ran', 'Duration', 'GapInRun']

excel =pd.read_csv("data/all_normed_logged6.csv")

# cols = excel[prioCols].columns.tolist()
# y = excel["CalcPrio"].to_numpy()
# X = excel[prioCols].to_numpy()
# with open ("proritisation1.txt", "a") as f:
#             f.write(str(cols) + "\n")

# classify_all(X, y)
# classify_all(X, y, default=False)

# y = excel["CalcPrio"].to_numpy()
# X = excel[prioColsMore].to_numpy()
# cols = excel[prioColsMore].columns.tolist()
# name = "priotisiation_normed.txt"
# with open (name, "a") as f:
#         f.write(str(cols) + "\n")


# # classify_all(X, y, name)
# classify_all(X, y, name, default=False)



excel =pd.read_csv("data/all_normed_logged6.csv")
y = excel["CalcPrio"].to_numpy()
X = excel[prioColsMore].to_numpy()
cols = excel[prioColsMore].columns.tolist()
name = "more_normalised.txt"
with open (name, "a") as f:
            f.write(str(cols) + "\n")



# classify_all(X, y, name)
# classify_all(X, y, name, default=False)

# without_target_and_redundent = ["Name","Duration","Cycle","Month","Quarter","Failure_Percentage","GapInRun","InSameCycle","CycleRun","last_run","last_result","times_ran","gap_cats","my_maturity"]
import matplotlib.pyplot as plt

def graph():


    data = [
        "score = 0.8083610080093768, time = 737.8757054805756 [Gradient Boosted Classifier]",
        "score = 0.5686657550302794, time = 2240.974885702133 [SVM Classifier] params = {'C': 0.4, 'decision_function_shape': 'ovo', 'gamma': 'scale', 'kernel': 'poly'}", 
        "score = 0.8099238132447744, time = 1084.9372770786285 [Random Forest Classifer] params = {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 30}", 
        "score = 0.6046102754444227, time = 0.13751983642578125 [Naive Bayes Classifier] params = {}", 
        "score = 0.7129181929181929, time = 67.31144309043884 [Enemble1]",
        "score = 0.7966788766788767, time = 27.295373678207397 [Enemble2]"
    ]
    data2 = [
        "score = 0.798788825942567, time = 14.150893688201904 [Gradient Boosted Classifier] params = {}",
        "score = 0.426645829263528, time = 28.25215196609497 [SVM Classifier] params = {}",
        "score = 0.7993748779058409, time = 5.750375509262085 [Random Forest Classifer] params = {}",
        "score = 0.6143778081656573, time = 0.46706318855285645 [Naive Bayes Classifier] params = {}",
        "score =  0.7177045177045177, time = 27.6362783908844 [Enemble1]",
        "score = 0.8043956043956044, time = 10.65587043762207 [Ensemble2]",
    ]

    data3 = [
        "score = 0.798788825942567, time = 14.150893688201904 [Gradient Boosted Classifier] params = {}",
        "score = 0.426645829263528, time = 28.25215196609497 [SVM Classifier] params = {}",
        "score = 0.7993748779058409, time = 5.750375509262085 [Random Forest Classifer] params = {}",
        "score = 0.6143778081656573, time = 0.46706318855285645 [Naive Bayes Classifier] params = {}",
        "score =  0.7177045177045177, time = 27.6362783908844 [Enemble1]",
        "score = 0.8043956043956044, time = 10.65587043762207 [Ensemble2]"
    ]
    
    data1 = [
        "score = 0.9816370384840789, time = 844.172189950943 [Gradient Boosted Classifier] params {'learning_rate': 0.2, 'loss': 'log_loss', 'n_estimators': 270}", 
        "score = 0.5690564563391287, time = 1478.1122269630432 [SVM Classifier] params {'C': 1.6, 'decision_function_shape': 'ovo', 'gamma': 'scale', 'kernel': 'poly'}", 
        "score = 0.9835905450283259, time = 385.283056974411 [Random Forest Classifer] params {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 60}",
        "score = 0.614573158820082, time = 0.12885355949401855 [Naive Bayes Classifier] params = {}", 
        "score = 0.8265286188708733, time = 26.32169270515442 [Ensemble1]",
        "score =  0.9663996874389529, time = 8.901706218719482 [Ensemble2]"
    ]

    # Extracting relevant data
    classifiers = []
    scores = []
    times = []


    for entry in data1:
        if "score" in entry:
#            input()
            score = float(entry.split("score = ")[1].split(",")[0])
            scores.append(score)
            
            print(score)
            
            time = float(entry.split("time = ")[1].split(" ")[0])
            times.append(time)
            
            print(time)
            
            classifier = entry.split("[")[1].split("]")[0]
            classifiers.append(classifier)
            
            print(classifier)
        elif "Accuracy" in entry:
            score = float(entry.split("Accuracy: ")[1].split(",")[0])
            scores.append(score)
            
            time = float(entry.split("time = ")[1].split(" ")[0])
            times.append(time)
            
            classifiers.append("Cross Validation")

    # Plotting the bar chart for scores
    bar_width = 0.35
    index = np.arange(len(classifiers))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting bars for scores
    bars1 = ax1.bar(index, scores, bar_width, label='Scores', color='tab:blue')

    ax1.set_xlabel('Classifiers')
    ax1.set_ylabel('Scores')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Classifier Performance')

    # Adding labels to the bars for scores
    for bar, score in zip(index, scores):
        ax1.text(bar, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)

    # Creating a secondary y-axis for time
    ax2 = ax1.twinx()

    # Plotting bars for times
    bars2 = ax2.bar(index + bar_width, times, bar_width, label='Time', color='tab:orange')

    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Adding labels to the bars for times
    for bar, time in zip(index + bar_width, times):
        ax2.text(bar, time, f'{time:.4f}', ha='center', va='bottom', fontsize=9)

    # Setting x-axis ticks and labels
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(classifiers, rotation=45, ha='right')

    # Adding legend
    fig.legend(loc='upper right', bbox_to_anchor=(0.8, 1))


    plt.tight_layout()
    plt.show()
    
    
input()
graph()


import matplotlib.pyplot as plt
import numpy as np
def graph2():

    data = [
        "score = 0.9816370384840789, time = 844.172189950943 [Gradient Boosted Classifier] params = {'learning_rate': 0.2, 'loss': 'log_loss', 'n_estimators': 270}", 
        "score = 0.5690564563391287, time = 1478.1122269630432 [SVM Classifier] params = {'C': 1.6, 'decision_function_shape': 'ovo', 'gamma': 'scale', 'kernel': 'poly'}", 
        "score = 0.9835905450283259, time = 385.283056974411 [Random Forest Classifier] params = {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 60}",
        "score = 0.614573158820082, time = 0.12885355949401855 [Naive Bayes Classifier] params = {}", 
        "score = 0.8265286188708733, time = 26.32169270515442 [Cross Validation] children = ['Gradient Boosted Classifier', 'SVM Classifier', 'Random Forest Classifier', 'Naive Bayes Classifier']",
        "score =  0.9663996874389529, time = 8.901706218719482 [Cross Validation] children = ['Gradient Boosted Classifier', 'Random Forest Classifier']"
    ]

    # Extracting relevant data
    classifiers = []
    scores = []
    times = []
    params = []

    for entry in data:
        if "score" in entry:
            score = float(entry.split("score = ")[1].split(",")[0])
            scores.append(score)
            
            time = float(entry.split("time = ")[1].split(" ")[0])
            times.append(time)
            
            classifier = entry.split("[")[1].split("]")[0]
            classifiers.append(classifier)

            # Extract params if available
            param_start = entry.find("params = {")
            if param_start != -1:
                param_str = entry[param_start:]
                param_str = param_str.split("}")[0] + "}"
                params.append(param_str)
            else:
                params.append(None)

    # Plotting the bar chart for scores
    bar_width = 0.35
    index = np.arange(len(classifiers))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting bars for scores
    bars1 = ax1.bar(index, scores, bar_width, label='Scores', color='tab:blue')

    ax1.set_xlabel('Classifiers')
    ax1.set_ylabel('Scores')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_title('Classifier Performance')

    # Adding labels to the bars for scores
    for bar, score, param in zip(index, scores, params):
        ax1.text(bar, score, f'{score:.4f}', ha='center', va='bottom', fontsize=9)
        if param:
            ax1.text(bar, score - 0.05, param, ha='center', va='bottom', fontsize=7, wrap=True)

    # Creating a secondary y-axis for time
    ax2 = ax1.twinx()

    # Plotting bars for times
    bars2 = ax2.bar(index + bar_width, times, bar_width, label='Time', color='tab:orange')

    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Adding labels to the bars for times
    for bar, time in zip(index + bar_width, times):
        ax2.text(bar, time, f'{time:.4f}', ha='center', va='bottom', fontsize=9)

    # Setting x-axis ticks and labels
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(classifiers, rotation=45, ha='right')

    # Adding legend
    fig.legend(loc='upper left', bbox_to_anchor=(0.8, 1))

    # Adding table for parameters
    table_data = []
    for classifier, param in zip(classifiers, params):
        table_data.append([classifier, param if param else "N/A"])

    column_labels = ["Classifier", "Parameters"]
    ax1.table(cellText=table_data, colLabels=column_labels, loc='bottom', cellLoc='center')

    plt.tight_layout()
    plt.show()


# graph2()