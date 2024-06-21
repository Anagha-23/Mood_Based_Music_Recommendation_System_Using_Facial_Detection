import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from fer import Video, FER
from moviepy.editor import *
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

# function: submit video
def submit_video():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    open_file = filedialog.askopenfilenames()
    clip= VideoFileClip(open_file[0]).subclip(0,10)
    clip.write_videofile('movie.mp4')
    clip.close()

# function: load and process video
def load_and_process():
    face_detector = FER(mtcnn=True)
    input_video = Video('movie.mp4')
    processing_data = input_video.analyze(face_detector, display=False)    
    vid_df = input_video.to_pandas(processing_data)
    vid_df = input_video.get_first_face(vid_df)
    vid_df = input_video.get_emotions(vid_df)
    return vid_df
  
vid_df = load_and_process()

# function: def top emo/ 2 emo || same and opposite feeling depending on desired use case

def top_emo(vid_df): # to improve, add flex to define num of emotions vs hard coding
    # compute weights 
    emo_matrix=pd.DataFrame(vid_df.sum(axis=0))
    emo_matrix.columns=['emo_score']
    emo_matrix['weight']=100*emo_matrix['emo_score']/np.sum(emo_matrix['emo_score'])
    
    # determine top emotion or n emotions
    max_emo_label=emo_matrix['weight'].idxmax()
    max_emo_value=np.max(emo_matrix['weight'])
    emo_threshold=85
    feeling=()
    if max_emo_value>=emo_threshold:
        feeling=max_emo_label
    else:
        emo_matrix.sort_values('weight',inplace=True,ascending=False)
        emo_matrix=emo_matrix.reset_index()
        first = emo_matrix['index'][0].title()
        second = emo_matrix['index'][1].title()
        feeling = tuple(sorted((first,second)))

  # hard code same and diff feeling songs

    spotify_moods = ["Distressed", "Energetic", "Excitement", "Contentment","Depressed","Misery"]
    video_moods = ["Angry", "Disgust", "Fear", "Happy","Sad","Surprised","Neutral"]
    dict_of_video_moods_same = {}
    dict_of_video_moods_diff = {}

    ## same-feeling songs

    ### singles
    dict_of_video_moods_same[('Angry','Angry')] = ['Distressed',"Energetic"]
    dict_of_video_moods_same[('Disgust','Disgust')] = ['Distressed']
    dict_of_video_moods_same[('Fear','Fear')] = ['Distressed','Misery']
    dict_of_video_moods_same[('Happy','Happy')] = ['Excitement']
    dict_of_video_moods_same[('Sad','Sad')] = ['Depressed']
    dict_of_video_moods_same[('Surprised','Surprised')] = ['Energetic']
    dict_of_video_moods_same[('Neutral','Neutral')] = ['Depressed','Excitement']

    ### doubles
    for i in range(len(video_moods)):
        for j in range(i+1,len(video_moods)):
            dict_of_video_moods_same[tuple(sorted((video_moods[i],video_moods[j])))] = []

    #### angry
    dict_of_video_moods_same[('Angry','Disgust')] = ['Misery']
    dict_of_video_moods_same[('Angry','Fear')] = ['Distressed','Energetic']
    dict_of_video_moods_same[('Angry','Happy')] = ['Energetic']
    dict_of_video_moods_same[('Angry','Sad')] = ['Depressed']
    dict_of_video_moods_same[('Angry','Surprised')] = ['Distressed']
    dict_of_video_moods_same[('Angry','Neutral')] = ['Distressed','Energetic']

    #### disgust
    dict_of_video_moods_same[('Disgust','Fear')] = ['Distressed']
    dict_of_video_moods_same[('Disgust','Happy')] = ['Excitement']
    dict_of_video_moods_same[('Disgust','Sad')] = ['Misery']
    dict_of_video_moods_same[('Disgust','Surprised')] = ['Energetic']
    dict_of_video_moods_same[('Disgust','Neutral')] = ['Distressed']

    #### fear
    dict_of_video_moods_same[('Fear','Happy')] = ['Energetic']
    dict_of_video_moods_same[('Fear','Sad')] = ['Depressed','Misery']
    dict_of_video_moods_same[('Fear','Surprised')] = ['Misery']
    dict_of_video_moods_same[('Fear','Neutral')] = ['Distressed','Misery']

    #### happy
    dict_of_video_moods_same[('Happy','Sad')] = ['Contentment','Misery']
    dict_of_video_moods_same[('Happy','Surprised')] = ['Contentment','Excitement']
    dict_of_video_moods_same[('Happy','Neutral')] = ['Excitement']

    #### sad
    dict_of_video_moods_same[('Sad','Surprised')] = ['Distressed']

    #### neutral
    dict_of_video_moods_same[('Neutral','Sad')] = ['Depressed']
    dict_of_video_moods_same[('Neutral','Surprised')] = ['Energetic']

    ## opposite-feeling songs

    dict_of_video_moods_diff[('Angry','Angry')] = ['Contentment']
    dict_of_video_moods_diff[('Disgust','Disgust')] = ['Contentment']
    dict_of_video_moods_diff[('Fear','Fear')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Happy','Happy')] = ['Excitement']
    dict_of_video_moods_diff[('Sad','Sad')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Surprised','Surprised')] = ['Contentment']
    dict_of_video_moods_diff[('Neutral','Neutral')] = ['Excitement','Depressed']  

    ### doubles
    for i in range(len(video_moods)):
        for j in range(i+1,len(video_moods)):
            dict_of_video_moods_diff[tuple(sorted((video_moods[i],video_moods[j])))] = []

    #### angry
    dict_of_video_moods_diff[('Angry','Disgust')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Angry','Fear')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Angry','Happy')] = ['Contentment']
    dict_of_video_moods_diff[('Angry','Sad')] = ['Contentment','Energetic']
    dict_of_video_moods_diff[('Angry','Surprised')] = ['Contentment']
    dict_of_video_moods_diff[('Angry','Neutral')] = ['Contentment']

    #### disgust
    dict_of_video_moods_diff[('Disgust','Fear')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Disgust','Happy')] = ['Contentment']
    dict_of_video_moods_diff[('Disgust','Sad')] = ['Contentment','Energetic']
    dict_of_video_moods_diff[('Disgust','Surprised')] = ['Contentment']
    dict_of_video_moods_diff[('Disgust','Neutral')] = ['Contentment']

    #### fear
    dict_of_video_moods_diff[('Fear','Happy')] = ['Contentment']
    dict_of_video_moods_diff[('Fear','Sad')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Fear','Surprised')] = ['Contentment']
    dict_of_video_moods_diff[('Fear','Neutral')] = ['Contentment','Excitement']

    #### happy - intentionally the same recommendations to same-feeling
    dict_of_video_moods_diff[('Happy','Sad')] = ['Contentment','Misery']
    dict_of_video_moods_diff[('Happy','Surprised')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Happy','Neutral')] = ['Excitement']

    #### sad
    dict_of_video_moods_diff[('Sad','Surprised')] = ['Energetic']

    #### neutral
    dict_of_video_moods_diff[('Neutral','Sad')] = ['Contentment','Excitement']
    dict_of_video_moods_diff[('Neutral','Surprised')] = ['Contentment']
    
    # return same and opposite feeling list
    same=dict_of_video_moods_same[feeling]
    opposite=dict_of_video_moods_diff[feeling]
    return same, opposite

## Include top_emo later
#spotify_category('valence_arousal_dataset.csv', top_emo(vid_df)[0], top_emo(vid_df)[1])
mood_centroids = {'Depressed':[0.19066008, 0.21390282],'Misery':[0.29176613, 0.55255357], 'Contentment': [0.67930098, 0.46349509], 'Distressed':[0.21726171, 0.89302689], 'Excitement':[0.83923924, 0.80557057], 'Energetic':[0.53893319, 0.83543712]}

def top10_combined_moods(df, mood1, mood2):
    data = pd.read_csv(df)    
    mean_valence = np.average([mood_centroids[mood1][0], mood_centroids[mood2][0]])
    mean_energy = np.average([mood_centroids[mood1][1], mood_centroids[mood2][1]])
    selected_moods = data.loc[data['moods_label'].isin([mood1, mood2])]
    selected_moods['euclidian_distance'] = ((selected_moods['valence'] - mean_valence)**2 + (selected_moods['energy'] - mean_energy)**2)**(1/2)
    #Select 1 song from 10 songs that have closest Euclidian Distance
    selected_track = selected_moods.sort_values('euclidian_distance').iloc[0:10]['track_name'].sample(n=10)
    return selected_track

submit_video()
len(top_emo(vid_df)[1])

def button_click(selected_mood):
    data = pd.read_csv('valence_arousal_dataset_labeled.csv')
    feelings = top_emo(vid_df)
    if selected_mood == 'same' and len(feelings[0]) > 1:
        return top10_combined_moods('valence_arousal_dataset_labeled.csv', feelings[0][0], feelings[0][1])
    elif selected_mood == 'opposite' and len(feelings[1]) > 1:
        return top10_combined_moods('valence_arousal_dataset_labeled.csv', feelings[1][0], feelings[1][1])
    elif selected_mood == 'same':
        return data[data['moods_label'] == feelings[0][0]].sample(n=10)['track_name']
    else:
        return data[data['moods_label'] == feelings[1][0]].sample(n=10)['track_name']
        
button_click('opposite')
