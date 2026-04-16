import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Normalise a raw SMS message into a clean token string for TF-IDF.

    Pipeline (matches dissertation section 3.4):
      1. Lowercase
      2. Remove URLs, email addresses, and phone numbers (they vary too much
         across messages to be reliable lexical features — s3.4.2)
      3. Remove punctuation and all non-alphabetic characters (s3.4.3)
      4. Tokenise with NLTK word_tokenize (s3.4.4)
      5. Remove English stop words (s3.4.5)
      6. Lemmatise to root forms (s3.4.6)
      7. Drop single-character tokens (s3.4.7)
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)        # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # remove email addresses
    text = re.sub(r"\+?\d[\d\s\-().]{7,}\d", " ", text)  # remove phone numbers
    text = re.sub(r"[^a-z\s]", " ", text)                 # keep only letters
    text = re.sub(r"\s+", " ", text).strip()              # collapse whitespace

    tokens = word_tokenize(text)
    # Drop stop-words (e.g. "the", "is") and single characters, then reduce each
    # word to its base form so "calling" and "called" count as the same feature.
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    return " ".join(tokens)


def preprocess_series(series):
    return series.apply(clean_text)
