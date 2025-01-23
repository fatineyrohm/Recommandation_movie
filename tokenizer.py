import nltk
import unicodedata
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus.reader.wordnet import VERB, NOUN, ADJ, ADV
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class LemmatizerTokenizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def get_wordnet_pos(self, word, pos_tag):
        tag = pos_tag[0].upper()
        tag_dict = {"J": ADJ, "N": NOUN, "V": VERB, "R": ADV}
        return tag_dict.get(tag, NOUN)

    def __call__(self, doc):
        doc = unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8')
        tokens = word_tokenize(doc.lower())
        pos_tags = pos_tag(tokens)
        return [self.lemmatizer.lemmatize(t, self.get_wordnet_pos(t, pos)) for t, pos in pos_tags if t not in self.stop_words and t.isalpha()]


