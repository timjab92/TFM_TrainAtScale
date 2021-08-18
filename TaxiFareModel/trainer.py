import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data, clean_data, df_optimized
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from google.cloud import storage
from TaxiFareModel import params

MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "first_experiment"


class Trainer(object):
    params = dict(
        nrows=1000,
        upload=True,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
        gridsearch=False,
        optimize=True,
        estimator="xgboost",
        mlflow=True,  # set to True to log params to mlflow
        experiment_name=EXPERIMENT_NAME,
        pipeline_memory=None,  # None if no caching and True if caching expected
        distance_type="manhattan",
        feateng=[
            "distance_to_center", "direction", "distance", "time_features",
            "geohash"
        ],
        n_jobs=-1)  # Try with njobs=1 and njobs = -1

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = EXPERIMENT_NAME

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ], memory=self.params)
        time_pipe = Pipeline(
            [('time_enc', TimeFeaturesEncoder('pickup_datetime')),
             ('ohe', OneHotEncoder(handle_unknown='ignore'))],
            memory=self.params)
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                "pickup_latitude",
                "pickup_longitude",
                'dropoff_latitude',
                'dropoff_longitude'
            ]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([('preproc', preproc_pipe),
                                  ('linear_model', LinearRegression())],
                                 memory=self.params)

    def run(self):
        self.set_pipeline()
        self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    # def save_model(self):
    #     """Save the model into a .joblib format"""
    #     joblib.dump(self.pipeline, 'model.joblib')
    #     print(colored("model.joblib saved locally", "green"))


    # def upload_model_to_gcp():

    #     client = storage.Client()

    #     bucket = client.bucket(params.BUCKET_NAME)

    #     blob = bucket.blob(params.STORAGE_LOCATION)

    #     blob.upload_from_filename('model.joblib')


    def save_model(self):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        #        upload_model_to_gcp()
        client = storage.Client()
        bucket = client.bucket(params.BUCKET_NAME)
        blob = bucket.blob(params.STORAGE_LOCATION)
        blob.upload_from_filename('model.joblib')

        print(
            f"uploaded model.joblib to gcp cloud storage under \n => {params.STORAGE_LOCATION}"
        )




    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)




if __name__ == "__main__":
    # Get and clean data
    df = get_data()
    df = clean_data(df)
    df = df_optimized(df, verbose=True)
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and
    trainer = Trainer(X=X_train, y=y_train)
    trainer.set_experiment_name('xp2')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    print(f"rmse: {rmse}")
    trainer.save_model()
