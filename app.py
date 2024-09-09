from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import requests

app = Flask(__name__)

# Load your data (Alumni List)
df = pd.read_excel('Alumni List.xlsx')

# Initialize TfidfVectorizer
tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',
                      token_pattern=r'\w{1,}', ngram_range=(1, 3), stop_words='english')

# Fill missing 'skills' in the data
def fillna(row):
    for i in range(3):
        if row.isna()['skills'] and len(df[df['position'] == row['position']]) > i:
            row['skills'] = df[df['position'] == row['position']]['skills'].iloc[i]
            return row['skills']
        else:
            return row['skills']

df['skills'] = df.apply(fillna, axis=1)
df = df[~df['skills'].isna()]

# Collect data from LinkedIn using Proxycurl
def collect_data(url):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    api_key = 'jk22A0HBk_3GBcme6Oh7fQ'
    headers = {'Authorization': 'Bearer ' + api_key}

    response = requests.get(api_endpoint, params={'url': url, 'skills': 'include'}, headers=headers)
    profile_data = response.json()
    return profile_data

# Recommendation function
def recommend_people(url):
    profile_data = collect_data(url)
    a = ','.join(i.lower() for i in profile_data['skills'])
    tfv_matrix_skills = tfv.fit_transform(df['skills'])
    sig = sigmoid_kernel(tfv_matrix_skills, tfv.transform([a]))
    cos = cosine_similarity(tfv_matrix_skills, tfv.transform([a]))
    score = 0.7 * cos.flatten() + 0.3 * sig.flatten()
    df['score'] = score
    result = df.sort_values('score', ascending=False)
    return result[['fullname', 'linkedin']].iloc[:10]

# Define the API route for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    linkedin_url = data.get('url')
    if not linkedin_url:
        return jsonify({'error': 'LinkedIn URL is required'}), 400

    try:
        recommendations = recommend_people(linkedin_url)
        return recommendations.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Home route for testing
@app.route('/')
def home():
    return "Recommendation API is running"

if __name__ == '__main__':
    app.run(debug=True)
