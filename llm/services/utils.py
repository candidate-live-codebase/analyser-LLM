from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def remove_stop_words(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    return " ".join(filtered_tokens)
