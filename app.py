from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess text
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Function to fetch content from URL
def fetch_content_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return preprocess_text(content)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return ""

# Function to search the web for similar content
def search_web(query, num_results=10):
    urls = []
    try:
        for url in search(query, num_results=num_results):
            urls.append(url)
    except Exception as e:
        print(f"Error in web search: {e}")
    return urls

# Function to calculate similarity using Sentence-BERT
def calculate_similarity(original_text, fetched_texts):
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    fetched_embeddings = [model.encode(text, convert_to_tensor=True) for text in fetched_texts]
    cosine_similarities = [util.cos_sim(original_embedding, fetched_embedding).item() for fetched_embedding in fetched_embeddings]
    
    # Normalize the similarities so that the total adds up to 100
    total_cosine_similarity = sum(cosine_similarities)
    if total_cosine_similarity > 0:
        cosine_similarities = [(sim / total_cosine_similarity) * 100 for sim in cosine_similarities]
    
    return cosine_similarities

# Function to extract text from an uploaded file
def extract_text_from_file(file):
    return file.read().decode('utf-8')

# Route to handle form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    plagiarism_results = []
    check_message = ""
    total_similarity = 0
    input_text = ""
    urls = []
    similarities = []

    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.txt'):
                input_text = extract_text_from_file(file)
        
        # If text is not uploaded from a file, use textarea input
        if not input_text:
            input_text = request.form['input_text'].strip()

        # Validate input text and word count
        if not input_text:
            check_message = "Please provide input text or upload a document."
        else:
            word_count = len(word_tokenize(input_text))
            if word_count < 10:
                check_message = "The statement contains less than 10 words. No plagiarism check required."
            else:
                # Preprocess the input text
                preprocessed_text = preprocess_text(input_text)
                
                # Search for similar content on the web
                urls = search_web(preprocessed_text, num_results=10)
                
                # Fetch content from the URLs
                fetched_texts = [fetch_content_from_url(url) for url in urls if url]
                
                # Calculate similarity percentages using Sentence-BERT
                similarities = calculate_similarity(preprocessed_text, fetched_texts)
                
                # Prepare a list of tuples (URL, similarity)
                results = [(urls[i], similarities[i]) for i in range(len(urls))]
                
                # Sort results in descending order by similarity
                results = sorted(results, key=lambda x: x[1], reverse=True)
                
                # Prepare the results for display
                for url, similarity in results:
                    plagiarism_results.append(f"URL: {url} : {similarity:.2f}%")
                
                # Total similarity is 100% since we normalized the individual similarities
                total_similarity = 100

    return render_template(
        'index.html',
        plagiarism_results=plagiarism_results,
        total_similarity=total_similarity,
        check_message=check_message
    )

if __name__ == '__main__':
    app.run(debug=True)
