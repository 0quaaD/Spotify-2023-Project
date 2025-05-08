import pickle
import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


with open('./model/spotify_2023_random_forest.pkl','rb') as file:
    model_forest = pickle.load(file)

with open('./model/spotify_2023_gradient_boosting.pkl','rb') as file:
    model_gradient = pickle.load(file)

with open('./model/test_X.pkl','rb') as file:
    x_test = pickle.load(file)

with open('./model/test_y.pkl','rb') as file:
    y_test = pickle.load(file)

with open('./model/scaler.pkl','rb') as file:
    scaler = pickle.load(file)

def load_data():
    return pd.read_csv('./dataset/spotify-2023.csv',on_bad_lines = 'warn', encoding='latin1')
def format_streams(n):
    if(n >= 1e+9):
        return f"{n / 1e+9:.2f}B"
    elif(n >= 1e+6):
        return f"{n / 1e+6:.2f}M"
    elif(n >= 1e+3):
        return f"{n / 1e+3}K"
    else:
        return str(n)

data = load_data()
st.title("🎧 Spotify 2023 ML Dashboard")

st.header("📊 The Dataset")
st.write("This dataset contains a comprehensive list of the most famous songs of 2023 as listed on Spotify. The dataset offers a wealth of features beyond what is typically available in similar datasets. It provides insights into each song's attributes, popularity, and presence on various music platforms. The dataset includes information such as 'track name', 'artist(s) name', 'release date,' 'Spotify playlists and charts', 'streaming statistics', 'Apple Music presence', 'Deezer presence', 'Shazam charts', and various audio features.")

if st.checkbox("The Entire Dataset"):
    st.write(data.head(10))

st.write("🎶 Popularity Prediction")
features = ['in_deezer_charts','in_spotify_charts',
         'in_apple_charts','in_deezer_playlists',
         'in_apple_playlists','in_spotify_playlists']


# Best-case scenario input for testing
features_best_case = {
    'in_deezer_charts': 1,
    'in_spotify_charts': 1,
    'in_apple_charts': 1,
    'in_deezer_playlists': 1,
    'in_apple_playlists': 1,
    'in_spotify_playlists': 1
}
input_df_ = pd.DataFrame([features])
user_input = {feature: st.slider(feature, float(data[feature].min()), float(data[feature].max())) for feature in features}
input_df = pd.DataFrame([user_input])

if(st.button("Predict")):
    prediction_rf = model_forest.predict(input_df)[0]
    prediction_gd = model_gradient.predict(input_df)[0]
    st.subheader("🔍 Model Predictions")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Random Forest Prediction for # of Streams: ", format_streams(prediction_rf))
    with col2:
        st.metric("Gradient Boosting Prediction for # of Streams: ", format_streams(prediction_gd))

st.header("📈 Random Forest and Gradient Boosting Model's Results")
st.markdown("Those are the model's performance graphs. It can looks like there aren't much difference between those plots. However When I compared the $R^2$ Score's of the both models, the Gradient Boosing Model gave the better $R^2$ score. Also the models use the same features from the dataset.", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    st.image('./model/plt_gradient_boosting_model.png',caption='Gradient Boosting Model', use_container_width=True)
with col4:
    st.image('./model/plt_random_forest_model.png',caption='Random Forest Model', use_container_width=True)


st.markdown("As you can see, the Gradient Boosting model performed better than the Random Forest model which it tells us that Gradient Boosting method is way better for our model with the given data.")
col5, col6 = st.columns(2)
with col5:
    st.image('./model/plt_random_forest_importance_bar_plot.png',caption='Random Forest Feature Importance Plot')
with col6:
    st.image('./model/plt_grad_boost_feature_importance.png', caption='Gradient Boosting Feature Importance Plot')

st.markdown("If you take a look at closely to the both Feature Importance plots, you would see that there are no difference between the models what they needed for training and testing.")

st.header("🏁 Conclusion")
st.markdown("""**In this project, we explored a huge dataset of Spotify's hit songs of 2023 and attempted to forecast their popularity based on features of different music platforms and audio features. We have trained and tested two ensemble models — Random Forest and Gradient Boosting — to evaluate their prediction abilities.

The results showed that the Gradient Boosting model outperformed the Random Forest model with a higher R2R2 value of 0.8679 compared to 0.8377. This indicates that Gradient Boosting was superior at capturing the complex relationships within the dataset.

Both models are trained on identical feature sets and so it was reasonable to make the comparison. Feature importance plots also provide insight into which fields (e.g., playlist status and platform charts) are playing most significant roles towards making the song popular. 

Lesson: In this case, Gradient Boosting is a better method than predicting song popularity, and perhaps could be optimised or worked further upon in the future with improved performance.**""")

