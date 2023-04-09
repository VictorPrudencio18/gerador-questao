from flask import Flask, render_template, request
import spacy
import nltk
from collections import defaultdict, Counter
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

nlp = spacy.load("en_core_web_trf")

def question_exists(question, questions):
    return any(q.lower() == question.lower() for q in questions)

from nltk.corpus import stopwords

def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    return ' '.join([word for word in nltk.word_tokenize(text) if (word.isalnum() or word in (',', '.', '?', '!', ';', ':', '-', '(', ')')) and word.lower() not in stop_words])

def generate_question_from_token(token):
    if token.dep_ in ('nsubj', 'nsubjpass'):
        return f"Who or what {token.head.lemma_} {token.text}?"
    elif token.dep_ == 'dobj':
        return f"What does {token.head.lemma_} involve in terms of {token.text}?"
    elif token.dep_ == 'prep':
        if token.head.dep_ in ('nsubj', 'nsubjpass', 'dobj'):
            return f"How are {token.head.head.lemma_} and {token.text} related?"
    elif token.dep_ == 'attr':
        return f"What is a characteristic feature of {token.text}?"
    elif token.dep_ == 'pobj' and token.head.dep_ == 'prep':
        if token.head.head.dep_ in ('nsubj', 'nsubjpass', 'dobj'):
            return f"How are {token.head.head.lemma_} and {token.text} related?"
    elif token.dep_ == 'advmod':
        if token.head.pos_ == 'VERB':
            return f"How does {token.text} affect the meaning of {token.head.lemma_}?"
    else:
        return None


def generate_questions(doc, entities):
    questions = []
    for sent in doc.sents:
        for token in sent:
            question = generate_question_from_token(token)
            if question and not question_exists(question, questions):
                questions.append(question)

    for label, texts in entities.items():
        for text in texts:
            if label == "DATE":
                question = f"What events took place on {text}?"
            elif label == "ORG":
                question = f"What are the objectives of {text}?"
            elif label == "NP":
                question = None
                token = nlp(text)[0]
                if token.pos_ in ('NOUN', 'PROPN'):
                    question = f"Talk aboult the importance of {text}"
                elif token.pos_ in ('ADJ', 'ADV'):
                    question = f"Talk aboult the {text}  and your impact in the subject matter"
                elif token.pos_ == 'VERB':
                    question = f"What results from {text}?"

            if question and not question_exists(question, questions):
                questions.append(question)
    return questions

from sklearn.metrics.pairwise import cosine_similarity

def rank_questions(questions, text, top_n=15):
    # Tokenize o texto e as perguntas
    text_words = nltk.word_tokenize(text)
    question_words = [nltk.word_tokenize(question) for question in questions]

    # Calcule o TF-IDF para o texto e as perguntas
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([" ".join(text_words)] + [" ".join(words) for words in question_words])
    text_tfidf = tfidf_matrix[0]

    # Função para calcular o escore das perguntas
    def score(question_tfidf):
        # Use a similaridade de cosseno entre o texto e a pergunta como escore
        return cosine_similarity(text_tfidf, question_tfidf)[0][0]

    # Classifique as perguntas com base em seus escores
    question_scores = [score(tfidf_matrix[i]) for i in range(1, tfidf_matrix.shape[0])]
    ranked_question_indices = heapq.nlargest(top_n, range(len(question_scores)), key=lambda i: question_scores[i])

    # Retorne as perguntas classificadas
    return [questions[i] for i in ranked_question_indices]



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form["text"]
        text = preprocess_text(text)
        doc = nlp(text)

        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        for chunk in doc.noun_chunks:
            entities["NP"].append(chunk.text)

        questions = generate_questions(doc, entities)
        top_questions = rank_questions(questions, text)
        return render_template("index.html", questions=top_questions)

    return render_template("index.html")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8000)