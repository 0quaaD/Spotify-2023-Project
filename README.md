# 🎧 Spotify 2023 Hit Songs Popularity Prediction
This project utilizes machine learning models to predict the popularity of Spotify's hit songs in 2023 based on various audio and platform attributes. The models used are Random Forest and Gradient Boosting, which were trained and tested to ascertain their performance in predicting song popularity.

## Table of Contents  
- Project Overview
- The Dataset
- Technologies used (languages and tools)
- Features
- Installation
- Usage
- Model Evaluation
- Contributions
- License

## Project Overview
In this project, we dealt with a large dataset of the most popular songs of Spotify in 2023. We wanted to predict the popularity of these songs using various features, such as audio features (e.g., tempo, danceability, energy) and platform-specific data (e.g., playlist status, chart positions).

We trained and tested two machine learning models:

- Random Forest: An ensemble model based on bagging.

- Gradient Boosting: A boosting-based ensemble model.

We evaluated the models after training them with the R² score to determine how accurately each model could predict song popularity. We also conducted a feature importance analysis to determine which features contribute the most to a song's popularity.

## 📊 Dataset
The [dataset](dataset/spotify-2023.csv) used for this project is complete information on Spotify's top hit songs for the year 2023. It is a combination of audio features extracted from Spotify's API and platform-specific metadata such as playlist status and chart position.  

1st 5 rows  
| track_name | artist(s)_name | artist_count | released_year | released_month | released_day | in_spotify_playlists | in_spotify_charts | streams | in_apple_playlists | in_apple_charts | in_deezer_playlists | in_deezer_charts | in_shazam_charts | bpm | key | mode | danceability_% | valence_% | energy_% | acousticness_% | instrumentalness_% | liveness_% | speechiness_% |
|----------|----------|----------|----------|---------|---------|----------|----------|----------|----------|---------|---------|----------|----------|----------|----------|---------|---------|----------|----------|----------|----------|---------|---------|
| Seven (feat. Latto) (Explicit Ver.) | Latto, Jung Kook | 2 | 2023 | 7 | 14 | 553 | 147 | 141381703 | 43 | 263 | 45 | 10 | 826 | 125 | B | Major | 80 | 89 | 83 | 31 | 0 | 8 | 4 |
| LALA | Myke Towers | 1 | 2023 | 3 | 23 | 1474 | 48 | 133716286 | 48 | 126 | 58 | 14 | 382 | 92 | C# | Major | 71 | 61 | 74 | 7 | 0 | 10 | 4 |
| vampire | Olivia Rodrigo | 1 | 2023 | 6 | 30 | 1397 | 113 | 140003974 | 94 | 207 | 91 | 14 | 949 | 138 | F | Major | 51 | 32 | 53 | 17 | 0 | 31 | 6 |
| Cruel Summer | Taylor Swift | 1 | 2019 | 8 | 23 | 7858 | 100 | 800840817 | 116 | 207 | 125 | 12 | 548 | 170 | A | Major | 55 | 58 | 72 | 11 | 0 | 11 | 15|
| WHERE SHE GOES |	Bad Bunny |	1 |	2023 |	5 |	18 |	3133 |	50 |	303236322 |	84 |	133 |	87 |	15 |	425 |	144 |	A |	Minor |	65 |	23 |	80 |	14 |	63 |	11 |	6 |



### Key components of the dataset  

- **Audio Features (from Spotify API)**:  
  - `danceability`: How suitable a track is for dancing.
  - `energy` :  Intensity and activity level of a track.
  - `key` : The estimated overall key of the track.
  - `loudness` : Overall loudness in decibels (dB).
  - `mode` : Major or minor scale.
  - `speechiness` : Presence of spoken words.
  - `acousticness` :  Likelihood the track is acoustic.
  - `instrumentalness` : Predicts whether a track contains vocals.
  - `liveness` : Detects the presence of an audience.
  - `valence` : Musical positivity.
  - `tempo` : Speed or pace of the track in BPM.
- **Platform Metadata**:
  - `in_spotify_charts` : Whether the song appeared on Spotify Charts.
  - `in_apple_charts` : Whether the song appeared on Apple Music Charts.
  - `in_deezer_charts` : Whether the song appeared on Deezer Charts.
  - `in_shazam_charts` : Whether the song appeared on Shazam Charts.
  - `in_spotify_playlists` : Presence in Spotify editorial/user playlists.
### ℹ️ Source of the Dataset
The dataset appears to have been compiled from multiple streaming and analytics platforms, likely through web scraping or API access.
The original source of the dataset is : [Spotify 2023 Data](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023)
If you want to see the details of the dataset I would certainly recommend the website.

## 🤖 Technologies used (Languages and tools)
- **Python 3.x**
- **Pandas** and **Numpy** for data manipulation
- **Scikit-learn** for machine learning models (Random Forest, Gradient Boosting)
- **Matplotlib** and **Seaborn** for visualizations
- **Streamlit** for web app interface (if applicable)

## Features
- **Data Processing**: Cleaned and preprocessed a large dataset of Spotify's 2023 hit songs.
- **Model Training**: Trained machine learning models to predict song popularity.
- **Model Comparison**: Evaluated and compared the performance of Random Forest and Gradient Boosting models.
- **Feature Importance**: Analyzed which features were most important in predicting popularity.
- **Web App** (if applicable): Visualized results using Streamlit for user-friendly interactions.

## 👨‍💻 Installation 
If you want to run this project on your local machine, then you should install the packages on the [requirements.txt](requirements.txt)  

**1.**: Clone the repo  

```
git clone https://github.com/0quaaD/Spotify-2023-Project.git
---- OR ----
git clone git@github.com:0quaaD/Spotify-2023-Project.git
---- THEN ----
cd Spotify-2023-Project
```

**2.** : Create a virtual environment
```
python3 -m venv myenv
```
**3.** : Acivate the environment
- On macOS/linux:
  ```
  source venv/bin/activate
  ```
- On Windows:
  ```
  venv\Scripts\activate
  ```
**4.** : Install the required tools, libraries
```
pip install -r requirements.txt
```

## Usage
Once you have the every tools for the project, then you can run the project:  

**1.** : Run the project on the **Jupyter Notebook**
```
jupyter-notebook
```
 **2.** :  Run the project on **Streamlib**
```
streamlit run app.py
```
**3.** : See the model decleration and plotting the graphs
- You can execute this command to train, test the model and plot the results 
  ``` 
  python3 ./model/model_declaration.py
  ```

## Model Evaluation  
The performance of the models was compared using the R² score. Summary of results:
- **Gradient Boosting**: Had an R² score of 0.8679, meaning that it was better at capturing complex relationships in the data.
- **Random Forest**: Had an R² score of 0.8377.
Feature importance calculation showed that platform-related features (e.g., playlist status, chart positions) were the crucial determinants that determined the popularity of the songs.

## ⌛ Contributions

Contributions are welcome! If you’d like to contribute to this project, feel free to fork the repository, create a new branch, and submit a pull request.

Please make sure to follow the standard Git workflow and provide meaningful commit messages.
