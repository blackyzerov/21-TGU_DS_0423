import os

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

img = Image.open("streamlit.png")
st.image(img, width=700)

st.markdown(
    "<h1 style='text-align: center; color: black;'>Сервис рекомендации фильмов по содержанию</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("Как пользоваться сервисом!")  # Toolbar left
st.sidebar.info(
    """
    1. Выберите жанры фильмов,
    2. Выберите годы выпуска фильмов в прокат,
    3. Введите или выберите фильм, который вам нравится,
    4. Нажмите кнопку для рекомендации! 
    """
)

selected_data = st.multiselect('Введите год фильмов:',recsys.get_data())  # """string to select by years"""

if selected_data: # """filter by years and TfidfVectorizer"""
    recsys.get_filter_data(selected_data)
    recsys.TfidfVectorizer()

selected_genres = st.multiselect('Введите жанр фильмов:',recsys.get_genres())  # string to select by genre

if selected_genres:  # """filter by genre and TfidfVectorizer"""
    recsys.get_filter_genres(selected_genres)
    recsys.TfidfVectorizer()

selected_movie = st.selectbox('Введите или выберите фильм, который вам нравится:',recsys.get_title())
# string to select by movie

if st.button('Нажми для рекомендаций'):   # recommendation button
    if selected_movie:
        recommended_movie_names = recsys.recommendation(
            selected_movie, top_k=TOP_K)
        if len(recommended_movie_names) == 0:
            st.write("""Нет рекомендаций по вашему выбору.
                         Измените выбор жанра или года.""")
        else:
            st.write("Рекомендованные фильмы:")
            recommended_movie_names = recsys.recommendation(selected_movie, top_k=TOP_K)
            recommended_movie_posters = omdbapi.get_posters(recommended_movie_names)
            movies_col = st.columns(len(recommended_movie_names))
            for index, col in enumerate(movies_col):
                with col:
                    st.image(recommended_movie_posters[index])
                    st.subheader(recommended_movie_names[index])

st.sidebar.info(
    """
    mail: blacky.belov@gmail.com
    mail: teodoroe@student.21-school.ru
    Intensive Parallel 21 TGU_DS_0423
    Tribe Tournament Uranus
    08.08.2023
    """
)
