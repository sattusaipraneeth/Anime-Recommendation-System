import sys
from anime_recommender.exception.exception import AnimeRecommendorException
from anime_recommender.loggers.logging import logging
from anime_recommender.utils.main_utils.utils import load_csv_data 
from anime_recommender.entity.artifact_entity import DataIngestionArtifact
from anime_recommender.model_trainer.top_anime_filtering import PopularityBasedFiltering
 
class PopularityBasedRecommendor: 
    """
    A class that provides anime recommendations based on different popularity criteria. 
    """
    def __init__(self,data_ingestion_artifact = DataIngestionArtifact):
        """
        Initializes the PopularityBasedRecommendor with the ingested anime dataset.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): An artifact containing the feature store file paths. 
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise AnimeRecommendorException(e,sys)
        
    def initiate_model_trainer(self,filter_type:str):
        """
        Trains the popularity-based recommender model and logs the top anime recommendations 
        based on the specified filter type.

        Args:
            filter_type (str): The type of filtering to apply. 
                                Options include:
                                    - 'popular_animes': Most popular anime based on user engagement.
                                    - 'top_ranked_animes': Highest ranked anime.
                                    - 'overall_top_rated_animes': Overall top-rated anime.
                                    - 'favorite_animes': Most favorited anime.
                                    - 'top_animes_members': Anime with the highest number of members.
                                    - 'popular_anime_among_members': Most popular anime among members.
                                    - 'top_avg_rated': Anime with the highest average ratings.  
        """
        try:
            logging.info("Loading transformed data...")
            df = load_csv_data(self.data_ingestion_artifact.feature_store_anime_file_path)

            recommender = PopularityBasedFiltering(df)

            if filter_type == 'popular_animes': 
                popular_animes = recommender.popular_animes(n =10) 
                logging.info(f"Popular Anime recommendations: {popular_animes}") 

            elif filter_type == 'top_ranked_animes': 
                top_ranked_animes = recommender.top_ranked_animes(n =10) 
                logging.info(f"top_ranked_animes recommendations: {top_ranked_animes}")  

            elif filter_type == 'overall_top_rated_animes': 
                overall_top_rated_animes = recommender.overall_top_rated_animes(n =10) 
                logging.info(f"overall_top_rated_animes recommendations: {overall_top_rated_animes}") 

            elif filter_type == 'favorite_animes': 
                favorite_animes = recommender.favorite_animes(n =10) 
                logging.info(f"favorite_animes recommendations: {favorite_animes}") 

            elif filter_type == 'top_animes_members': 
                top_animes_members = recommender.top_animes_members(n = 10) 
                logging.info(f"top_animes_members recommendations: {top_animes_members}") 

            elif filter_type == 'popular_anime_among_members': 
                popular_anime_among_members = recommender.popular_anime_among_members(n =10) 
                logging.info(f"popular_anime_among_members recommendations: {popular_anime_among_members}") 
            
            elif filter_type == 'top_avg_rated': 
                top_avg_rated = recommender.top_avg_rated(n =10) 
                logging.info(f"top_avg_rated recommendations: {top_avg_rated}") 

        except Exception as e:
            raise AnimeRecommendorException(e,sys)