from flask import Flask, render_template, request, redirect, url_for
import googleapiclient.discovery
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import pandas as pd
import time

app = Flask(__name__)

DEVELOPER_KEY = ""  #Add your actual YouTube Data API v3 developer key (free to make with a Google Account)

#Initialize sentiment analysis pipeline
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True)

label_mapping = {'LABEL_0': 'Negative', 'LABEL_1': 'Neutral', 'LABEL_2': 'Positive'}

def analyze_comments(video_id):
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEVELOPER_KEY)

    total_comments_to_fetch = 500
    comments_fetched = 0
    page_token = None

    comments_data = {'label': [], 'score': [], 'comment': []}

    while comments_fetched < total_comments_to_fetch:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, total_comments_to_fetch - comments_fetched),
            pageToken=page_token
        )
        
        response = request.execute()

        for item in response.get('items', []):
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            
            #Preprocess comment
            comment_words = []
            for word in comment_text.split(' '):
                if word.startswith('@') and len(word) > 1:
                    word = '@user'
                elif word.startswith('http'):
                    word = "http"
                comment_words.append(word)
            comment_proc = " ".join(comment_words)

            #Sentiment analysis
            scores = sentiment_pipeline(comment_proc)

            for score in scores:
                label_id = score['label']
                score_value = score['score']
                human_readable_label = label_mapping.get(label_id, 'Unknown')
                
                comments_data['label'].append(human_readable_label)
                comments_data['score'].append(score_value)
                comments_data['comment'].append(comment_text)

            comments_fetched += 1

        #Check if there are more comments to fetch
        page_token = response.get('nextPageToken')
        if not page_token:
            break

    return comments_data

def plot_sentiment_graph(comments_data):
    #Create a DataFrame from the data
    df = pd.DataFrame(comments_data)

    #Assuming df is your DataFrame
    df['dot_size'] = 8  #You can adjust this value to change the size of the dots

    fig = px.scatter(df, y="label", x="score", color="label", hover_data=["comment"],
                    labels={'label': 'Sentiment Category', 'score': 'Sentiment Score'},
                    title='Sentiment Analysis of YouTube Comments',
                    template='plotly_white',
                    size='dot_size',  
                    category_orders={'label': ['Negative', 'Neutral', 'Positive']})

    #Calculate total and mean for each category
    category_stats = df.groupby('label')['score'].agg(['count', 'mean']).reset_index()

    #Annotate with total and mean information
    for i, row in category_stats.iterrows():
        fig.add_annotation(
            x=row['mean'], 
            y=row['label'],
            text=f"Total: {row['count']}<br>Mean: {row['mean']:.2f}",
            showarrow=True,
            arrowhead=4,
            ax=0,
            ay=-40
        )

    #Customize layout
    fig.update_layout(
        xaxis_title='Sentiment Score',
        yaxis_title='Sentiment Category',
        showlegend = False
    )

    #Show the plot
    fig.show()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    link = request.form['youtube_link']

    #Extract video ID using string manipulation
    start_index = link.find("v=") + 2
    end_index = link.find("&", start_index)
    if end_index == -1:
        video_id = link[start_index:]
    else:
        video_id = link[start_index:end_index]

    if not video_id:
        return render_template('error.html', message='Invalid YouTube link. Please provide a valid link.')

    comments_data = analyze_comments(video_id)
    plot_sentiment_graph(comments_data)

    #Add a delay of 2 seconds (adjust as needed)
    time.sleep(2)

    #Redirect back to the main page after analyzing
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)