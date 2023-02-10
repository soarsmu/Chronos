import re
import spacy
from spacy.tokenizer import Tokenizer
from stopwords import ENG_STOPWORDS, KEYWORDS, REFERENCES, generate_stop_words

# file_name = ""
# REFERENCES = generate_stop_words(file_name, )

text = spacy.load("en_core_web_sm")
text.Defaults.stop_words.update(ENG_STOPWORDS, KEYWORDS)
stop_words = text.Defaults.stop_words
pattern = re.compile(r"[a-zA-Z][a-z]+")

def extended_is_stop(token):

    return token.is_stop or token.lower_ in stop_words or token.lemma_ in stop_words

def text_processor(docs):
    
    docs = [word.lemma_.lower() for word in text(str(pattern.findall(docs))) 
                            if not (extended_is_stop(word) or word.is_punct)]                       
    return docs