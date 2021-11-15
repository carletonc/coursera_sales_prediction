import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error




def get_data(path='tmp/data/train.pkl'):
    if isinstance(path, str):
        TRAIN = pd.read_pickle(path).fillna(0)
    elif isinstance(path, pd.DataFrame):
        TRAIN = path.copy()
    else:
        return "Not a valid input for `path`."
    
    train, val = train_test_split(TRAIN, test_size=0.3, random_state=42, shuffle=True)
    
    X_train = TRAIN.iloc[train.index].drop('item_cnt_month', axis=1).copy()
    y_train = TRAIN.iloc[train.index]['item_cnt_month'].copy()
    X_val = TRAIN.iloc[val.index].drop('item_cnt_month', axis=1).copy()
    y_val = TRAIN.iloc[val.index]['item_cnt_month'].copy()
    return (X_train, y_train, X_val, y_val)



def train_experiment(X_train, y_train, pipeline, params, n_experiments=5, random_state=42):
    search = RandomizedSearchCV(pipeline, 
                                param_distributions=params, 
                                n_iter=n_experiments,
                                cv=5, 
                                scoring='neg_mean_squared_error',
                                refit=True,
                                n_jobs=-1,
                                random_state=random_state)
    search.fit(X_train, y_train)
    
    print('Best Score: ', np.abs(search.best_score_))
    print('Best Params: ', search.best_estimator_)
    return search



def regression_report(target, prediction, name=None):
    mse = mean_squared_error(target, prediction)
    rmse = mean_squared_error(target, prediction, squared=False)
    mae = mean_absolute_error(target, prediction)
    r2 = r2_score(target, prediction)
    #r2_adj = r2_score(target, prediction, multioutput='variance_weighted')
    #max_er = max_error(target, prediction)
    
    if name is not None:
        print(f'\n=== {name} ===')
    else:
        print()
    print('\t\tMean Squared Error: \t\t', mse)
    print('\t\tRoot Mean Squared Error: \t', rmse)
    print('\t\tMean Absolute Error: \t\t', mae)
    print('\t\tR2: \t\t\t\t', r2)
    #print('\t\tR2 Adjusted: \t\t\t', r2_adj)
    #print('\t\tMax Error: \t\t\t', max_er)
    return (mse, rmse, mae, r2) # r2_adj, max_er



def log_experiment(pipeline, train_report, val_report):
    with mlflow.start_run():
        pipeline_steps = pipeline.named_steps
        pipeline_step_keys = list(pipeline_steps.keys())
        for key in pipeline_step_keys:
            mlflow.log_param(key, pipeline_steps[key])

        mlflow.log_param('pipeline', pipeline)

        mlflow.log_metric('train_mse', train_report[0])
        mlflow.log_metric('train_rmse', train_report[1])
        mlflow.log_metric('train_mae', train_report[2])
        mlflow.log_metric('train_r2', train_report[3])
        #mlflow.log_metric('train_r2_adj', train_report[4])
        #mlflow.log_metric('train_max_error', train_report[5])

        mlflow.log_metric('val_mse', val_report[0])
        mlflow.log_metric('val_rmse', val_report[1])
        mlflow.log_metric('val_mae', val_report[2])
        mlflow.log_metric('val_r2', val_report[3])
        #mlflow.log_metric('val_r2_adj', val_report[4])
        #mlflow.log_metric('val_max_error', val_report[5])

        mlflow.sklearn.log_model(pipeline, "model")
    
    
def run_experiment(data, pipeline, params, n_experiments):
    X_train, y_train, X_val, y_val = data
    # train model
    search = train_experiment(X_train, y_train, pipeline, params, n_experiments=n_experiments)
    best_pipeline = search.best_estimator_
    # best_pipeline is already re-fit, we dont need to fit again
    # best_pipeline.fit(X_train, y_train) 
    
    # acquire training metrics
    y_train_pred = best_pipeline.predict(X_train)
    train_report = regression_report(y_train, y_train_pred, name='TRAIN')
    
    # acquire validation metrics
    y_val_pred = best_pipeline.predict(X_val)
    val_report = regression_report(y_val, y_val_pred, name='VALIDATION')
    
    log_experiment(best_pipeline, train_report, val_report)
    return
    

    
def main(pipeline=None, params=None, n_experiments=None, pipelines=None):
    # get data
    data = get_data()
    
    if isinstance(models, list):
        for i, model in enumerate(models):
            pipeline = model['pipeline']
            params = model['params']
            n_experiments = model['n_experiments']
            
            run_experiment(data, pipeline, params, n_experiments)
        return
            
    else:
        run_experiment(data, pipeline, params, n_experiments)
        return