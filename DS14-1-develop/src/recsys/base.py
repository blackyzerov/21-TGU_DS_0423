import pandas as pd
from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .utils import parse
import streamlit as st

@st.cache_data
def _load_base(path: str, index_col: str = 'id') -> pd.DataFrame:
    """Load CSV file to pandas.DataFrame"""
    return pd.read_csv(path, index_col=index_col)

class ContentBaseRecSys:

    def __init__(self, movies_dataset_filepath: str, distance_filepath: str):
        self.distance = _load_base(distance_filepath, index_col='id')
        self.distance.index = self.distance.index.astype(int)
        self.distance.columns = self.distance.columns.astype(int)
        self._init_movies(movies_dataset_filepath)

    def _init_movies(self, movies_dataset_filepath) -> None:
        self.movies = _load_base(movies_dataset_filepath, index_col='id')
        self.movies.index = self.movies.index.astype(int)
        self.movies['genres'] = self.movies['genres'].apply(parse)

    def get_title(self) -> List[str]:
        return self.movies['title'].values

    def get_genres(self) -> Set[str]:
        genres = [item for sublist in self.movies['genres'].values.tolist() for item in sublist]
        return set(genres)

    def get_data(self) -> Set[int]:
        return self.movies['date'].sort_values().dropna().unique().astype(int)

    def get_filter_data(self, selected_data: int):
        self.movies = self.movies.query('date in @selected_data')

    def get_filter_genres(self, selected_genres: str):
        self.movies = self.movies[self.movies.genres1.str.contains('|'.join(selected_genres), na=False)]

    def TfidfVectorizer(self):
        vec = TfidfVectorizer(stop_words='english', max_features=10000)
        matrix = vec.fit_transform(self.movies.combi)
        cosine_sim = cosine_similarity(matrix, matrix)
        self.distance = pd.DataFrame(cosine_sim, index=self.movies.index, columns=self.movies.index)

    def recommendation(self, title: str, top_k: int = 5) -> List[str]:
        """
        Returns the names of the top_k most similar movies with the movie "title"
        """
        indices = pd.Series(
            self.movies.index, index=self.movies['title']).drop_duplicates()

        idx = indices[title]
        sim_scores = list(enumerate(self.distance[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_k + 1]
        movie_ind = [i[0]for i in sim_scores]
        return list(self.movies['title'].iloc[movie_ind])
