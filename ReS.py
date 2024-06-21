# Importing Libraries
import streamlit as st

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Set the background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-photo/movie-background-collage_23-2149876014.jpg");
background-size: 100%;
}}
</style>
"""

# Inject the custom CSS into the Streamlit app
st.markdown(page_bg_img, unsafe_allow_html=True)



st.title('Movies Recommender System')
st.image("C:/EXTRA FILE/Downloads/output-onlinegiftools1.gif" , width=1400)

# ---------------------------------Now Macking Recommendations ------------------------------------

import pandas as pd
import numpy as np
import requests
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Add
from tensorflow.keras.optimizers import Adam

# .\new_venv\Scripts\activate

# Custom objects (if any)
custom_objects = {
    'Adam': Adam
}


# Working Data Frame
df = pd.read_csv("C:/Users/ANKUR KUMAR/Data Science/Projects/ML/Recommender System/Data.csv")

# Loading Model
model = load_model('big_movies_model.h5', custom_objects=custom_objects)

# -----------------Data Preprocessing and Prediction---------------------------
# Preprocess the data
user_ids = df['UserID'].unique().tolist()
movie_titles = df['Titles'].unique().tolist()

# Create mappings
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
movie_to_index = {title: index for index, title in enumerate(movie_titles)}

# Map the original ids to indices
df['user_index'] = df['UserID'].map(user_to_index)
df['movie_index'] = df['Titles'].map(movie_to_index)

# Define the number of users and movies
num_users = len(user_ids)
num_movies = len(movie_titles)

# Model Buildings

embedding_dim = 50 # You can adjust this

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# User embedding
user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_dim, input_length=1)(user_input)
user_vec = Flatten()(user_embedding)

# Movie embedding
movie_input = Input(shape=(1,))
movie_embedding = Embedding(num_movies, embedding_dim, input_length=1)(movie_input)
movie_vec = Flatten()(movie_embedding)

# Dot product of user and movie embeddings
dot_user_movie = Dot(axes=1)([user_vec, movie_vec])

# Prepare the inputs for the model
train_user_indices = train['user_index'].values
train_movie_indices = train['movie_index'].values
train_ratings = train['Ratings'].values

test_user_indices = test['user_index'].values
test_movie_indices = test['movie_index'].values
test_ratings = test['Ratings'].values

#---------------------------- Function to get recommendations for a user---------------------------------------
def recommend_movies(user_id, num_recommendations=5):
    user_idx = user_to_index[user_id]
    user_array = np.array([user_idx] * num_movies)
    movie_array = np.array(range(num_movies))

    predictions = model.predict([user_array, movie_array])
    predicted_ratings = predictions.flatten()

    movie_indices_sorted = predicted_ratings.argsort()[-num_recommendations:][::-1]
    recommended_titles = [movie_titles[i] for i in movie_indices_sorted]

    return recommended_titles

# ----------------Gettings Tables-------------------
UserID = df['UserID'].drop_duplicates()
UI = UserID.values
#-----------------Getting Tables----------------------

st.title('Recommend Movies for a User')
Select_UserID = st.selectbox("Select UserID to whom you want to recommend",UI)

user_id = Select_UserID
recommendations = recommend_movies(user_id)
X = pd.DataFrame(recommendations, columns=["Recommended Movies"])

# ----------------------Now Making a Good Recommendations-----------------
if st.button('Show Recommendation'):
    st.table(X)