import os
import sys 
import pandas as pd
import joblib
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException
from anime_recommender.constant import * 
from huggingface_hub import HfApi, HfFolder

def export_data_to_dataframe(dataframe: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """
    Saves a given Pandas DataFrame to a CSV file.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame should be stored.
    
    Returns:
        pd.DataFrame: The same DataFrame that was saved.
    """
    try:
        logging.info(f"Saving DataFrame to file: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        dataframe.to_csv(file_path, index=False, header=True)
        logging.info(f"DataFrame saved successfully to {file_path}.")
        return dataframe
    except Exception as e:
        logging.error(f"Error saving DataFrame to {file_path}: {e}")
        raise AnimeRecommendorException(e, sys)

def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a Pandas DataFrame.
    
    Args:
        file_path (str): The file path of the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        logging.info(f"Loading CSV data from file: {file_path}")
        df = pd.read_csv(file_path)
        logging.info("CSV file loaded successfully.")
        return df 
    except Exception as e:
        logging.error(f"Error loading CSV file {file_path}: {e}")
        raise AnimeRecommendorException(e, sys) from e

def save_model(model: object, file_path: str) -> None:
    """
    Saves a machine learning model to a file using joblib.
    
    Args:
        model (object): The model object to be saved.
        file_path (str): The file path where the model should be stored.
    """
    try:
        logging.info("Entered the save_model method.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            joblib.dump(model, file_obj)
        logging.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving model to {file_path}: {e}")
        raise AnimeRecommendorException(e, sys) from e

def load_object(file_path: str) -> object:
    """
    Loads a model or object from a file using joblib.
    
    Args:
        file_path (str): The file path of the saved model.
    
    Returns:
        object: The loaded model.
    """
    try:
        logging.info(f"Attempting to load object from {file_path}")
        if not os.path.exists(file_path):
            error_msg = f"The file: {file_path} does not exist."
            logging.error(error_msg)
            raise Exception(error_msg)
        with open(file_path, "rb") as file_obj:
            logging.info("Object loaded successfully.")
            return joblib.load(file_obj)
    except Exception as e:
        logging.error(f"Error loading object from {file_path}: {e}")
        raise AnimeRecommendorException(e, sys) from e
    
def upload_model_to_huggingface(model_path: str, repo_id: str, filename: str):
    """
    Uploads a trained model file to the specified Hugging Face repository.

    Args:
        model_path (str): Local path of the trained model file.
        repo_id (str): Hugging Face repository ID (e.g., 'krishnaveni76/anime-recommendation-models').
        filename (str): Name of the file when uploaded to Hugging Face.
    """
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model" 
        )
        logging.info(f"Successfully uploaded {filename} to Hugging Face: {repo_id}")
    except Exception as e:
        logging.info(f"Error uploading model to Hugging Face: {str(e)}")
        raise AnimeRecommendorException(e, sys) from e