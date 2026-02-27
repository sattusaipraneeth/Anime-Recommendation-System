import sys
import numpy as np
import pandas as pd 
from anime_recommender.loggers.logging import logging
from anime_recommender.exception.exception import AnimeRecommendorException

class PopularityBasedFiltering:
    """
    A recommender system that filters popular animes based on different criteria such as popularity, rank,
    average rating, number of members, and favorites.
    """
    def __init__(self, df):
        """
        Initialize the PopularityBasedFiltering class with a DataFrame.
        """
        try:
            logging.info("Initializing PopularityBasedFiltering class")
            self.df = df
            self.df['average_rating'] = pd.to_numeric(self.df['average_rating'], errors='coerce')
            self.df['average_rating'].fillna(self.df['average_rating'].median())
        except Exception as e:
            logging.error("Error initializing PopularityBasedFiltering: %s", str(e))
            raise AnimeRecommendorException(e, sys)
         
    def popular_animes(self, n=10):
        """
        Get the top N most popular animes.
        """
        logging.info("Fetching top %d most popular animes", n)
        sorted_df = self.df.sort_values(by=['popularity'], ascending=True)
        top_n_anime = sorted_df.head(n)
        return self._format_output(top_n_anime)
    
    def top_ranked_animes(self, n=10):
        """
        Get the top N ranked animes.
        """
        logging.info("Fetching top %d ranked animes", n)
        self.df['rank'] = self.df['rank'].replace('UNKNOWN', np.nan).astype(float)
        df_filtered = self.df[self.df['rank'] > 1]
        sorted_df = df_filtered.sort_values(by=['rank'], ascending=True)
        top_n_anime = sorted_df.head(n)
        return self._format_output(top_n_anime)
    
    def overall_top_rated_animes(self, n=10):
        """
        Get the top N highest-rated animes.
        """
        logging.info("Fetching top %d highest-rated animes", n)
        sorted_df = self.df.sort_values(by=['average_rating'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return self._format_output(top_n_anime)
    
    def favorite_animes(self, n=10):
        """
        Get the top N most favorited animes.
        """
        logging.info("Fetching top %d most favorited animes", n)
        sorted_df = self.df.sort_values(by=['favorites'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return self._format_output(top_n_anime)
    
    def top_animes_members(self, n=10):
        """
        Get the top N animes based on the number of members.
        """
        logging.info("Fetching top %d animes based on number of members", n)
        sorted_df = self.df.sort_values(by=['members'], ascending=False)
        top_n_anime = sorted_df.head(n)
        return self._format_output(top_n_anime)
    
    def popular_anime_among_members(self, n=10):
        """
        Get the top N animes popular among members based on the highest number of members and ratings.
        """
        logging.info("Fetching top %d popular animes among members", n)
        sorted_df = self.df.sort_values(by=['members', 'average_rating'], ascending=[False, False]).drop_duplicates(subset='name') 
        popular_animes = sorted_df.head(n)
        return self._format_output(popular_animes)
    
    def top_avg_rated(self, n=10): 
        """
        Get the top N highest-rated animes after handling missing values.
        """
        logging.info("Fetching top %d highest average-rated animes", n)
        self.df['average_rating'] = pd.to_numeric(self.df['average_rating'], errors='coerce')
        median_rating = self.df['average_rating'].median()
        self.df['average_rating'].fillna(median_rating)
        top_animes = (
            self.df.drop_duplicates(subset='name')
                    .nlargest(n, 'average_rating')[['name', 'average_rating', 'image url', 'genres']]
        )
        return self._format_output(top_animes)
    
    def _format_output(self, anime_df):
        """
        Format the output as a DataFrame with selected anime attributes.
        """
        return pd.DataFrame({
            'Anime name': anime_df['name'].values,
            'Image URL': anime_df['image url'].values,
            'Genres': anime_df['genres'].values,
            'Rating': anime_df['average_rating'].values
        })