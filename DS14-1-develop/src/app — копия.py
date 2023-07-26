import os

import time
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
from api.omdb import OMDBApi
from recsys import ContentBaseRecSys

TOP_K = 5
load_dotenv()

API_KEY = os.getenv("API_KEY")
MOVIES = os.getenv("MOVIES")
DISTANCE = os.getenv("DISTANCE")

omdbapi = OMDBApi(API_KEY)


recsys = ContentBaseRecSys(
    movies_dataset_filepath=MOVIES,
    distance_filepath=DISTANCE,
)

progress_bar = st.progress(0)
progress_text = st.empty()
for i in range(101):
    time.sleep(0.1)
    progress_bar.progress(i)
    progress_text.text(f"Progress: {i}%")

img = Image.open("streamlit.png")
st.image(img, width=700)

st.markdown(
    "<h1 style='text-align: center; color: black;'>Сервис рекомендации фильмов</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Как пользоваться сервисом!")
st.sidebar.info(
    """
    1. Выберите жанры фильмов,
    2. Выберите годы выпуска фильмов в прокат,
    3. Введите или выберите фильм, который вам нравится,
    4. Нажмите кнопку для рекомендации! 
    """
)

selected_data = st.multiselect(
    'Введите год фильмов:',
    recsys.get_data()
)
if selected_data:
    recsys.get_filter_data(selected_data)
    recsys.TfidfVectorizer()


selected_genres = st.multiselect(
    'Введите жанр фильмов:',
    recsys.get_genres()
)
if selected_genres:
    recsys.get_filter_genres(selected_genres)
    recsys.TfidfVectorizer()


selected_movie = st.selectbox(
    'Введите или выберите фильм, который вам нравится:',
    recsys.get_title()
)

if st.button('Нажми для рекомендаций'):
    st.write("Рекомендованные фильмы:")
    recommended_movie_names = recsys.recommendation(selected_movie, top_k=TOP_K)
    recommended_movie_posters = omdbapi.get_posters(recommended_movie_names)
    movies_col = st.columns(len(recommended_movie_names))
    for index, col in enumerate(movies_col):
        with col:
            st.image(recommended_movie_posters[index])
            st.subheader(recommended_movie_names[index])