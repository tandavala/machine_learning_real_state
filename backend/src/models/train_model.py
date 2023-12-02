import argparse
import yaml
from urllib.parse import urlparse
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import mlflow
from mlflow.sklearn import log_model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def read_params(config_path):
    with open(config_path) as yaml_file:
        return yaml.safe_load(yaml_file)


def transformer(cat_features, num_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('le', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, cat_features),
            ('num', numeric_transformer, num_features),

        ]
    )


def preprocessing(config, df):
    df_names = config['raw_data_config']['df_name']
    target = config['raw_data_config']['target']
    categorical_features = [var for var in df_names if df[var].dtype == 'O']
    numerical_features = [
        var for var in df_names if var not in categorical_features]

    return transformer(categorical_features, numerical_features)


def build_pipeline(config, df):
    preprocessor = preprocessing(config, df)

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('svm_reg', SVR(kernel='linear')),
    ])


def accuracy_measures(y_test, predictions):
    svm_reg_mae = mean_absolute_error(y_test, predictions)
    svm_reg_mse = mean_squared_error(y_test, predictions)
    svm_reg_rmse = np.sqrt(svm_reg_mae)
    print('Accuracy Measures')
    print('----------------------', '\n')
    print('MAE: ', svm_reg_mae)
    print('MSE: ', svm_reg_mse)
    print('RMSE: ', svm_reg_rmse)
    return svm_reg_mae, svm_reg_mse, svm_reg_rmse


def get_feat_and_target(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def running_mlflow(config, xtrain, ytrain, xtest, ytest):
    mlflow_config = config['mlflow_config']
    remover_server_uri = mlflow_config['remote_server_uri']

    mlflow.set_tracking_uri(remover_server_uri)
    mlflow.set_experiment(mlflow_config['experiment_name'])

    with mlflow.start_run(run_name=mlflow_config['run_name']) as mlops_run:

        pipeline = build_pipeline(config, xtrain)
        pipeline.fit(xtrain, ytrain)
        predictions = pipeline.predict(xtest)

        svm_reg_mae, svm_reg_mse, svm_reg_rmse = accuracy_measures(
            ytest, predictions)

        mlflow.log_metric('MAE', svm_reg_mae)
        mlflow.log_metric('MSE', svm_reg_mse)
        mlflow.log_metric('RMSE', svm_reg_rmse)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(
                pipeline, 'model', registered_model_name=mlflow_config['registered_model_name'])
        else:
            mlflow.sklearn.load_model(pipeline, 'model')


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]

    train = pd.read_csv(train_data_path, sep=',')
    test = pd.read_csv(test_data_path, sep=',')
    X_train, y_train = get_feat_and_target(train, target)
    X_test, y_test = get_feat_and_target(test, target)

    running_mlflow(config, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', default='params.yml')
    parsed_args = args.parse_args()

    train_and_evaluate(config_path=parsed_args.config)
