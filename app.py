import pandas as pd
import streamlit as st
from anime_recommender.model_trainer.content_based_modelling import ContentBasedRecommender
from anime_recommender.model_trainer.collaborative_modelling import CollaborativeAnimeRecommender
from anime_recommender.model_trainer.top_anime_filtering import PopularityBasedFiltering
import joblib
from anime_recommender.constant import *
from huggingface_hub import hf_hub_download
from datasets import load_dataset

def run_app():
    """
    Initializes the Streamlit app, loads necessary datasets and models, 
    and provides a UI for anime recommendations based on three methods: 
    Content-Based, Collaborative, and Popularity-Based Filtering. 🎬🎮
    """

    # Set page configuration
    st.set_page_config(page_title="Anime Recommendation System", layout="wide")

    # Load datasets if not present in session state
    if "anime_data" not in st.session_state or "anime_user_ratings" not in st.session_state:
        # Load datasets from Hugging Face (assuming no splits)
        animedataset = load_dataset(ANIME_FILE_PATH, split=None)
        mergeddataset = load_dataset(ANIMEUSERRATINGS_FILE_PATH, split=None)

        # Convert the dataset to Pandas DataFrame
        st.session_state.anime_data = pd.DataFrame(animedataset["train"])
        st.session_state.anime_user_ratings = pd.DataFrame(mergeddataset["train"]) 

    # Load models only once
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = {} 
        # Load models
        st.session_state.models_loaded["cosine_similarity_model"] = hf_hub_download(MODELS_FILEPATH, MODEL_TRAINER_COSINESIMILARITY_MODEL_NAME)
        st.session_state.models_loaded["item_based_knn_model_path"] = hf_hub_download(MODELS_FILEPATH, MODEL_TRAINER_ITEM_KNN_TRAINED_MODEL_NAME)
        st.session_state.models_loaded["user_based_knn_model_path"] = hf_hub_download(MODELS_FILEPATH, MODEL_TRAINER_USER_KNN_TRAINED_MODEL_NAME)
        st.session_state.models_loaded["svd_model_path"] = hf_hub_download(MODELS_FILEPATH, MODEL_TRAINER_SVD_TRAINED_MODEL_NAME)

        # Load the models using joblib
        with open(st.session_state.models_loaded["item_based_knn_model_path"], "rb") as f:
            st.session_state.models_loaded["item_based_knn_model"] = joblib.load(f)

        with open(st.session_state.models_loaded["user_based_knn_model_path"], "rb") as f:
            st.session_state.models_loaded["user_based_knn_model"] = joblib.load(f)

        with open(st.session_state.models_loaded["svd_model_path"], "rb") as f:
            st.session_state.models_loaded["svd_model"] = joblib.load(f)

        print("Models loaded successfully!")

    # Access the data from session state
    anime_data = st.session_state.anime_data 
    anime_user_ratings = st.session_state.anime_user_ratings

    # # Display dataset info
    # st.write("Anime Data:")
    # st.dataframe(anime_data.head())
    
    # st.write("Anime User Ratings Data:")
    # st.dataframe(anime_user_ratings.head())
    
    # Access the models from session state
    cosine_similarity_model_path = hf_hub_download(MODELS_FILEPATH, MODEL_TRAINER_COSINESIMILARITY_MODEL_NAME)
    item_based_knn_model = st.session_state.models_loaded["item_based_knn_model"]
    user_based_knn_model = st.session_state.models_loaded["user_based_knn_model"]
    svd_model = st.session_state.models_loaded["svd_model"] 
    print("Models loaded successfully!")
        
    # Streamlit UI
    app_selector = st.sidebar.radio(
        "Select App", ("Content-Based Recommender", "Collaborative Recommender", "Top Anime Recommender")
    )

    # Content-Based Recommender App
    if app_selector == "Content-Based Recommender":
        st.title("Content-Based Recommendation System") 
        try:
            
            anime_list = anime_data["name"].tolist()
            anime_name = st.selectbox("Pick an anime..unlock similar anime recommendations..", anime_list) 

            # Set number of recommendations
            max_recommendations = min(len(anime_data), 100)
            n_recommendations = st.slider("Number of Recommendations", 1, max_recommendations, 10)

            # Inject custom CSS for anime name font size
            st.markdown(
                """
                <style>
                .anime-title {
                    font-size: 14px !important;
                    font-weight: bold;
                    text-align: center;
                    margin-top: 5px;
                }
                </style>
                """,
                unsafe_allow_html=True,
            ) 
            # Get Recommendations
            if st.button("Get Recommendations"):
                try:
                    recommender = ContentBasedRecommender(anime_data)
                    recommendations = recommender.get_rec_cosine(anime_name, n_recommendations=n_recommendations,model_path=cosine_similarity_model_path)

                    if isinstance(recommendations, str):
                        st.warning(recommendations)
                    elif recommendations.empty:
                        st.warning("No recommendations found.🧐")
                    else:
                        st.write(f"Here are the Content-based Recommendations for {anime_name}:") 
                        cols = st.columns(5)
                        for i, row in enumerate(recommendations.iterrows()):
                            col = cols[i % 5]
                            with col:
                                st.image(row[1]['Image URL'], use_container_width=True)
                                st.markdown(
                                    f"<div class='anime-title'>{row[1]['Anime name']}</div>",
                                    unsafe_allow_html=True,
                                )
                                st.caption(f"Genres: {row[1]['Genres']} | Rating: {row[1]['Rating']}") 
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

    elif app_selector == "Collaborative Recommender":
        st.title("Collaborative Recommender System 🧑‍🤝‍🧑💬")
        
        try:  
            # Sidebar for choosing the collaborative filtering method
            collaborative_method = st.sidebar.selectbox(
                "Choose a collaborative filtering method:", 
                ["SVD Collaborative Filtering", "User-Based Collaborative Filtering", "Anime-Based KNN Collaborative Filtering"]
            )

            # User input
            if collaborative_method == "SVD Collaborative Filtering" or collaborative_method == "User-Based Collaborative Filtering": 
                user_ids = anime_user_ratings['user_id'].unique()  
                user_id = st.selectbox("Select your MyAnimeList user ID to get anime recommendations based on similar users", user_ids) 
                n_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=50, value=10)
            elif collaborative_method == "Anime-Based KNN Collaborative Filtering": 
                anime_list = anime_user_ratings["name"].dropna().unique().tolist() 
                anime_name = st.selectbox("Pick an anime, and we'll suggest more titles you'll love", anime_list)
                n_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=50, value=10)
    
            # Get recommendations
            if st.button("Get Recommendations"):
                # Load the recommender
                recommender = CollaborativeAnimeRecommender(anime_user_ratings) 
                if collaborative_method == "SVD Collaborative Filtering": 
                    recommendations = recommender.get_svd_recommendations(user_id, n=n_recommendations, svd_model=svd_model)  
                elif collaborative_method == "User-Based Collaborative Filtering": 
                    recommendations = recommender.get_user_based_recommendations(user_id, n_recommendations=n_recommendations, knn_user_model=user_based_knn_model)
                elif collaborative_method == "Anime-Based KNN Collaborative Filtering":
                    if anime_name: 
                        recommendations = recommender.get_item_based_recommendations(anime_name, n_recommendations=n_recommendations, knn_item_model=item_based_knn_model)
                    else:
                        st.error("Invalid Anime Name. Please enter a valid anime title.")
                
                if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                    if len(recommendations) < n_recommendations:
                        st.warning(f"Oops...Only {len(recommendations)} recommendations available, fewer than the requested {n_recommendations}.")
                    st.write(f"Here are the {collaborative_method} Recommendations:") 
                    cols = st.columns(5)
                    for i, row in enumerate(recommendations.iterrows()):
                        col = cols[i % 5]
                        with col:
                            st.image(row[1]['Image URL'], use_container_width=True)
                            st.markdown(
                                f"<div class='anime-title'>{row[1]['Anime Name']}</div>",
                                unsafe_allow_html=True,
                            ) 
                            st.caption(f"Genres: {row[1]['Genres']} | Rating: {row[1]['Rating']}")
                else:
                    st.error("No recommendations found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

    elif app_selector == "Top Anime Recommender":
        st.title("Top Anime Recommender System 🔥")
        
        try: 
            popularity_method = st.sidebar.selectbox(
                "Choose a Popularity-Based Filtering method:",
                [
                    "Popular Animes",
                    "Top Ranked Animes",
                    "Overall Top Rated Animes",
                    "Favorite Animes",
                    "Top Animes by Members",
                    "Popular Anime Among Members",
                    "Top Average Rated Animes",
                ]
            )
            
            n_recommendations = st.slider("Number of Recommendations:", min_value=1, max_value=500 , value=10)
            
            if st.button("Get Recommendations"): 
                recommender = PopularityBasedFiltering(anime_data)
                
                # Get recommendations based on selected method
                if popularity_method == "Popular Animes":
                    recommendations = recommender.popular_animes(n=n_recommendations)
                elif popularity_method == "Top Ranked Animes":
                    recommendations = recommender.top_ranked_animes(n=n_recommendations)
                elif popularity_method == "Overall Top Rated Animes":
                    recommendations = recommender.overall_top_rated_animes(n=n_recommendations)
                elif popularity_method == "Favorite Animes":
                    recommendations = recommender.favorite_animes(n=n_recommendations)
                elif popularity_method == "Top Animes by Members":
                    recommendations = recommender.top_animes_members(n=n_recommendations)
                elif popularity_method == "Popular Anime Among Members":
                    recommendations = recommender.popular_anime_among_members(n=n_recommendations)
                elif popularity_method == "Top Average Rated Animes":
                    recommendations = recommender.top_avg_rated(n=n_recommendations)
                else:
                    st.error("Invalid selection. Please choose a valid method.")
                    recommendations = None
                
                # Display recommendations
                if isinstance(recommendations, pd.DataFrame) and not recommendations.empty:
                    st.write(f" Here are the Recommendations:")
                    cols = st.columns(5)
                    for i, row in recommendations.iterrows():
                        col = cols[i % 5]
                        with col:
                            st.image(row['Image URL'], use_container_width=True)
                            st.markdown(
                                f"<div class='anime-title'>{row['Anime name']}</div>",
                                unsafe_allow_html=True,
                            )
                            st.caption(f"Genres: {row['Genres']} | Rating: {row['Rating']}")
                else:
                    st.error("No recommendations found.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    run_app()