from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact: 
    feature_store_anime_file_path:str
    feature_store_userrating_file_path:str

@dataclass
class DataTransformationArtifact:
    merged_file_path:str

@dataclass
class CollaborativeModelArtifact:
    svd_file_path:str
    item_based_knn_file_path:str
    user_based_knn_file_path:str
 
@dataclass
class ContentBasedModelArtifact:
    cosine_similarity_model_file_path:str