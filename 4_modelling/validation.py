from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import pandas as pd

X_head_drop = ['fam_target_ori', 'rot_target_ori', 'tau_target_ori', 'par_target_ori',
               'fam_target_hor', 'rot_target_hor', 'tau_target_hor', 'par_target_hor',
               'fam_target_hor_clayton', 'fam_target_hor_frank', 'fam_target_hor_gaussian',
               'fam_target_hor_gumbel', 'fam_target_hor_indep', 'fam_target_hor_joe',
               'fam_target_hor_student', 'fam_target_ori_clayton', 'fam_target_ori_frank',
               'fam_target_ori_gaussian', 'fam_target_ori_gumbel', 'fam_target_ori_indep',
               'fam_target_ori_joe', 'fam_target_ori_student', 'year_hor', 'symbol', 'ric', 'year']
y_head_fam_hor = ['fam_target_hor_clayton', 'fam_target_hor_frank', 'fam_target_hor_gaussian',
                  'fam_target_hor_gumbel', 'fam_target_hor_indep', 'fam_target_hor_joe', 'fam_target_hor_student']
y_head_fam_ori = ['fam_target_ori_clayton', 'fam_target_ori_frank', 'fam_target_ori_gaussian',
                  'fam_target_ori_gumbel', 'fam_target_ori_indep', 'fam_target_ori_joe', 'fam_target_ori_student'] 
y_head_tau_hor = ['tau_target_hor']
y_head_tau_ori = ['tau_target_ori']


def cross_validation (data, model, target):  
    # classification metrics
    auc_train_annual = []
    auc_valid_annual = []
    acc_train_annual = []
    acc_valid_annual = []
    prec_train_annual = []
    prec_valid_annual = []
    rec_train_annual = []
    rec_valid_annual = []
    f1_train_annual =  []
    f1_valid_annual =  []
    
    # regression metrics
    mse_train_annual = []
    mse_valid_annual = []  
    mae_train_annual = []
    mae_valid_annual = []
    r2_train_annual = []
    r2_valid_annual = []
    
    for i in range(10):
        # train validation splits (10 years (9 + 1) window rolled forward over 20 years -> 10 folds)
        # print(i)
        train_subset = data[(data['year'] >= (2001+i)) & (data['year'] <= (2008+i))]
        # print('Test' + str(2001+i) + '' + str(2008+i))
        
        valid_subset = data[data['year'] == (2009+i)]
        # print('Valid' + str(2009+i))
        
        X_train = train_subset.drop(columns=X_head_drop)
        X_valid = valid_subset.drop(columns=X_head_drop)

        if target == 'fam_target_hor':
            y_train = train_subset[y_head_fam_hor]
            y_valid = valid_subset[y_head_fam_hor]
        elif target == 'fam_target_ori':
            y_train = train_subset[y_head_fam_ori]
            y_valid = valid_subset[y_head_fam_ori]
        elif target == 'tau_target_hor':
            y_train = train_subset[y_head_tau_hor]
            y_valid = valid_subset[y_head_tau_hor]
        elif target == 'tau_target_ori':
            y_train = train_subset[y_head_tau_ori]
            y_valid = valid_subset[y_head_tau_ori]
        else:
            print("Wrong target parameter handed to function")

        # model estimation: fit model on ith fold train_subset
        model.fit(X_train, y_train)

        # forecast: predict next observation 
        pred_train = model.predict(X_train)
        pred_valid = model.predict(X_valid)

        # calculate scores
        if target == 'fam_target_hor' or target == 'fam_target_ori':
            auc_train_annual.append(roc_auc_score(y_train, pred_train, multi_class='ovr', average='macro'))
            auc_valid_annual.append(roc_auc_score(y_valid, pred_valid, multi_class='ovr', average='macro'))
            acc_train_annual.append(accuracy_score(y_train, pred_train))
            acc_valid_annual.append(accuracy_score(y_valid, pred_valid))
            prec_train_annual.append(average_precision_score(y_train, pred_train))
            prec_valid_annual.append(average_precision_score(y_valid, pred_valid))
            rec_train_annual.append(recall_score(y_train, pred_train, average='macro'))
            rec_valid_annual.append(recall_score(y_valid, pred_valid, average='macro'))
            f1_train_annual.append(f1_score(y_train, pred_train, average='macro'))
            f1_valid_annual.append(f1_score(y_valid, pred_valid, average='macro'))
        else:
            mse_train_annual.append(mean_squared_error(y_train, pred_train))
            mse_valid_annual.append(mean_squared_error(y_valid, pred_valid))
            mae_train_annual.append( mean_absolute_error(y_train, pred_train))
            mae_valid_annual.append(mean_absolute_error(y_valid, pred_valid))
            r2_train_annual.append(r2_score(y_train, pred_train))
            r2_valid_annual.append(r2_score(y_valid, pred_valid))

    if target == 'fam_target_hor' or target == 'fam_target_ori':
        scores = {
            'auc_train': auc_train_annual,
            'auc_valid': auc_valid_annual,
            'acc_train': acc_train_annual,
            'acc_valid': acc_valid_annual,
            'prec_train': rec_train_annual,
            'prec_valid': prec_valid_annual,
            'rec_train': rec_train_annual,
            'rec_valid': rec_valid_annual,
            'f1_train': f1_train_annual,
            'f1_valid': f1_valid_annual
        }
    else:
        scores = {
            'mse_train': mse_train_annual,
            'mse_valid': mse_valid_annual,
            'mae_train': mae_train_annual,
            'mae_valid': mae_valid_annual,
            'r2_train': r2_train_annual,
            'r2_valid': r2_valid_annual
        }
        
    return scores

def performance_test_shifted (data, opt_model, target):
    # classification metrics
    auc_train_annual = []
    auc_test_annual = []
    acc_train_annual = []
    acc_test_annual = []
    prec_train_annual = []
    prec_test_annual = []
    rec_train_annual = []
    rec_test_annual = []
    f1_train_annual =  []
    f1_test_annual =  []
    
    # regression metrics
    mse_train_annual = []
    mse_test_annual = []  
    mae_train_annual = []
    mae_test_annual = []
    r2_train_annual = []
    r2_test_annual = []

    ts_mean_estim_annual = []
    ts_mean_true_annual = []


    for i in range (10):
        # train validation splits (10 years window rolled forward over 20 years -> 10 folds)
        # print(i)
        train = data[(data['year'] >= (2001+i)) & (data['year'] <= (2009+i))]
        # print('Train' + str(2001+i) + '' + str(2009+i))
        test = data[(data['year']) == (2010+i)]
        # print('Test' + str(2010+i))

        X_train = train.drop(columns=X_head_drop)
        X_test = test.drop(columns=X_head_drop)
        
        if target == 'fam_target_hor':
            y_train = train[y_head_fam_hor]
            y_test = test[y_head_fam_hor]
        elif target == 'fam_target_ori':
            y_train = train[y_head_fam_ori]
            y_test = test[y_head_fam_ori]
        elif target == 'tau_target_hor':
            y_train = train[y_head_tau_hor]
            y_test = test[y_head_tau_hor]
        elif target == 'tau_target_ori':
            y_train = train[y_head_tau_ori]
            y_test = test[y_head_tau_ori]
        else:
            print("Wrong target parameter handed to function")
        
        # fit model on train data
        opt_model.fit(X_train, y_train)

        # forecast on test data
        pred_train = opt_model.predict(X_train)
        pred_test = opt_model.predict(X_test)
            
        # calculate scores
        if target == 'fam_target_hor' or target == 'fam_target_ori':
            auc_train_annual.append(roc_auc_score(y_train, pred_train, multi_class='ovr', average='macro'))
            auc_test_annual.append(roc_auc_score(y_test, pred_test, multi_class='ovr', average='macro'))
            acc_train_annual.append(accuracy_score(y_train, pred_train))
            acc_test_annual.append(accuracy_score(y_test, pred_test))
            prec_train_annual.append(average_precision_score(y_train, pred_train))
            prec_test_annual.append(average_precision_score(y_test, pred_test))
            rec_train_annual.append(recall_score(y_train, pred_train, average='macro'))
            rec_test_annual.append(recall_score(y_test, pred_test, average='macro'))
            f1_train_annual.append(f1_score(y_train, pred_train, average='macro'))
            f1_test_annual.append(f1_score(y_test, pred_test, average='macro'))
            ts_mean_true_annual.append(y_test.mean())
            ts_mean_estim_annual.append(pd.DataFrame(pred_test).mean())
        else:
            mse_train_annual.append(mean_squared_error(y_train, pred_train))
            mse_test_annual.append(mean_squared_error(y_test, pred_test))
            mae_train_annual.append( mean_absolute_error(y_train, pred_train))
            mae_test_annual.append(mean_absolute_error(y_test, pred_test))
            r2_train_annual.append(r2_score(y_train, pred_train))
            r2_test_annual.append(r2_score(y_test, pred_test))
            ts_mean_true_annual.append(float(y_test.mean()))
            ts_mean_estim_annual.append(float(pred_test.mean()))

    if target == 'fam_target_hor' or target == 'fam_target_ori':
        scores = {
            'auc_train': auc_train_annual,
            'auc_test': auc_test_annual,
            'acc_train': acc_train_annual,
            'acc_test': acc_test_annual,
            'prec_train': rec_train_annual,
            'prec_test': prec_test_annual,
            'rec_train': rec_train_annual,
            'rec_test': rec_test_annual,
            'f1_train': f1_train_annual,
            'f1_test': f1_test_annual,
            'ts_mean_true': ts_mean_true_annual,
            'ts_mean_estim': ts_mean_estim_annual
        }
    else:
        scores = {
            'mse_train': mse_train_annual,
            'mse_test': mse_test_annual,
            'mae_train': mae_train_annual,
            'mae_test': mae_test_annual,
            'r2_train': r2_train_annual,
            'r2_test': r2_test_annual,
            'ts_mean_true': ts_mean_true_annual,
            'ts_mean_estim':ts_mean_estim_annual
        }

    return scores