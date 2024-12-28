import re  # lib for regex expression
from nltk.corpus import stopwords  # NLTK for NLP tasks
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def preprocessing_step(text):
    # LOWER TEXT
    text = text.lower()
    # REMOVE ANY SPECIAL CHARACTER
    text = re.sub('[^a-zA-Z]', ' ', text)
    # TOKENIZATION FOR THE STATEMENT -- "I Love You ???!!" -> "I,Love,You,?,?,?,!,!,......."
    tokens = word_tokenize(text)
    # REMOVE STOPWORDS
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # STEMMING PORTER STEMMER -> FAST & INACCURATE / LEMMATIZATION  -> SLOW & ACCURATE
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    stemmed_tokens = ' '.join(stemmed_tokens)
    return stemmed_tokens
