import sys
import re
import string
import os
import numpy as np
import codecs
import pickle

# From scikit learn that got words from:
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"])


def load_glove(filename):
    """
    Read all lines from the indicated file and return a dictionary
    mapping word:vector where vectors are of numpy `array` type.
    GloVe file lines are of the form:

    the 0.418 0.24968 -0.41242 0.1217 ...

    Ignore stopwords.
    """
    text = get_text(filename)
    lines = text.split('\n')
    word_vectors = dict()

    for line in lines:
        words = line.split(' ')
        if words[0] not in ENGLISH_STOP_WORDS and words[0] != '/':
            vector = [float(value) for value in words[1:]]
            word_vectors[words[0]] = vector

    return word_vectors

def filelist(root):
    """Return a fully-qualified list of filenames under the root directory."""
    allfiles = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            if '.DS_Store' not in name:
                allfiles.append(os.path.join(path, name))

    return allfiles

def get_text(filename):
    """
    Load and return the text of a text file, assuming latin-1 encoding.
    """
    with codecs.open(filename, encoding='latin-1', mode='r') as f:
        return f.read()

def words(text):
    """
    Given a string, return a list of words normalized as follows:
    - Lowercase all words
    - Replace specified characters with spaces
    - Split on space to get word list
    - Ignore words < 3 char long and English stop words
    """
    text = text.lower()
    text = re.sub("[" + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    words_list = [word for word in text.split(' ') if len(word) >= 3 and word not in ENGLISH_STOP_WORDS]
    return words_list

def split_title(text):
    """Given text, return title and the rest of the article."""
    lines = text.split('\n')
    title = lines[0]
    article = lines[1:]
    article_string = ''.join(article)
    return title, article_string

def load_articles(articles_dirname, gloves):
    """Load all .txt files under articles_dirname and return a table."""
    article_list = filelist(articles_dirname)
    load_list = []

    for filename in article_list:
        article_data = []
        article_data.append(filename)
        text = get_text(filename)
        title, article_text = split_title(text)
        article_data.append(title)
        article_data.append(article_text)
        article_data.append(doc2vec(article_text, gloves))
        load_list.append(article_data)

    return load_list

def doc2vec(text, gloves):
    """Return the word vector centroid for the text."""
    txt = words(text)
    n_words = 0
    vector_sum = np.zeros((300,), dtype=float)

    for word in txt:
        if word in gloves:
            n_words += 1
            vector_sum += gloves[word]

    if n_words > 0:
        return vector_sum / n_words
    else:
        return np.zeros((300,), dtype=float)

def distances(article, articles):
    """Compute the Euclidean distance from article to every other article."""
    dist_list = []

    for art in articles:
        dif = np.subtract(article[3], art[3])
        e_dist = np.linalg.norm(dif)
        dist_list.append((e_dist, art[0]))

    return dist_list

def recommended(article, articles, n):
    """Return top n articles closest to the article."""
    article_list = []
    dist_list = distances(article, articles)
    sorted_distances = sorted(dist_list)

    for distance, filename in sorted_distances[1:n+1]:
        topic_filename = '/'.join(filename.split('/')[-2:])
        article_list.append((topic_filename, distance))

    return article_list

def main():
    glove_filename = sys.argv[1]
    articles_dirname = sys.argv[2]

    gloves = load_glove(glove_filename)
    articles = load_articles(articles_dirname, gloves)

    with open('articles.pkl', 'wb') as f:
        pickle.dump(articles, f)
    rec_dict = dict()
    for article in articles[1:]:
        splits = article[0].split('/')
        topic_filename = '/'.join(splits[-2:])
        rec_dict[topic_filename] = recommended(article, articles, 5)

    with open('recommend.pkl', 'wb') as f:
        pickle.dump(rec_dict, f)

if __name__ == "__main__":
    main()

