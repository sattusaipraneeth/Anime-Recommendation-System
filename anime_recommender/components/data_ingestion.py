import sys
import pandas as pd
from datasets import load_dataset
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException
from anime_recommender.entity.config_entity import DataIngestionConfig
from anime_recommender.entity.artifact_entity import DataIngestionArtifact
from anime_recommender.utils.main_utils.utils import export_data_to_dataframe

class DataIngestion:
    """
    A class responsible for data ingestion in the anime recommender system.

    This class fetches data from Hugging Face datasets, converts it into pandas DataFrame format, 
    and exports the processed data to storage for further use in the pipeline.
    """
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration settings for data ingestion. 
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise AnimeRecommendorException(e, sys)

    def fetch_data_from_huggingface(self, dataset_path: str, split: str = None) -> pd.DataFrame:
        """
        Fetches a dataset from Hugging Face and converts it into a pandas DataFrame. 
        Args:
            dataset_path (str): The path to the Hugging Face dataset.
            split (str, optional): The dataset split to be fetched (e.g., 'train', 'test'). Defaults to None.

        Returns:
            pd.DataFrame: The dataset converted into a pandas DataFrame. 
        """
        try:
            logging.info(f"Fetching data from Hugging Face dataset: {dataset_path}")
            # Load dataset from Hugging Face
            dataset = load_dataset(dataset_path, split=split)

            # Convert dataset to pandas DataFrame
            df = pd.DataFrame(dataset['train'])

            # Log some information about the data
            logging.info(f"Shape of the dataframe: {df.shape}")
            logging.info(f"Column names: {df.columns}")
            logging.info(f"Preview of the DataFrame:\n{df.head()}")
            logging.info("Data fetched successfully from Hugging Face.")
            
            return df

        except Exception as e:
            logging.error(f"An error occurred while fetching data: {str(e)}")
            raise AnimeRecommendorException(e, sys)

    def ingest_data(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process, fetching datasets and saving them to the feature store. 
        Returns:
            DataIngestionArtifact: An artifact containing paths to the ingested datasets. 
        """
        try:
            # Load anime and rating data from Hugging Face datasets
            anime_df = self.fetch_data_from_huggingface(self.data_ingestion_config.anime_filepath)
            rating_df = self.fetch_data_from_huggingface(self.data_ingestion_config.rating_filepath)

            # Export data to DataFrame
            export_data_to_dataframe(anime_df, file_path=self.data_ingestion_config.feature_store_anime_file_path)
            export_data_to_dataframe(rating_df, file_path=self.data_ingestion_config.feature_store_userrating_file_path)

            # Create artifact to store data ingestion info
            dataingestionartifact = DataIngestionArtifact(
                feature_store_anime_file_path=self.data_ingestion_config.feature_store_anime_file_path,
                feature_store_userrating_file_path=self.data_ingestion_config.feature_store_userrating_file_path
            ) 
            return dataingestionartifact

        except Exception as e:
            raise AnimeRecommendorException(e, sys)
