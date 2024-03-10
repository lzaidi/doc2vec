# Launch with python app.py

from flask import Flask, render_template
import sys
import pickle

app = Flask(__name__)

@app.route("/")
def articles():
    """Show a list of article titles."""
    return render_template('articles.html', arts=articles)


@app.route("/article/<topic>/<filename>")
def article(topic, filename):
    """
    Show an article with a relative path filename. Assumes the BBC structure of
    topic/filename.txt so our URLs follow that.
    """
    address = f'{topic}/{filename}'
    return render_template('article.html', art_topic=topic, art_filename=filename, recs=recommended, arts=articles, adr=address)

# Load articles and recommendations from pickle files
with open('articles.pkl', 'rb') as f:
    articles = pickle.load(f)

with open('recommend.pkl', 'rb') as f:
    recommended = pickle.load(f)

# You may need more code here or not

# For local debug
if __name__ == '__main__':
    app.run(debug=True)
