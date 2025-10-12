# Importação de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2

files= ["treino/train_arcaico_moderno.csv",
        "treino/train_complexo_simples.csv",
        "treino/train_literal_dinamico.csv"]

file_names = {
    "treino/train_arcaico_moderno.csv": "Resumo Arcaico x Moderno",
    "treino/train_complexo_simples.csv": "Resumo Complexo x Simples",
    "treino/train_literal_dinamico.csv": "Resumo Literal x Dinâmico"
}

le = LabelEncoder()

tf_idf = TfidfVectorizer()

selector = SelectKBest(score_func=chi2, k=22000)

type_metrics = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'f1_macro'
]

model_nb = MultinomialNB()

# Leitura e pré-processamento
for file in files:
    df = pd.read_csv(file, sep=';')
    df = df.dropna(subset=['text'])
    Y = df['style']
    y_number = le.fit_transform(Y)
    X_tf = tf_idf.fit_transform(df['text'])
    X_new = selector.fit_transform(X_tf, y_number)

    results = cross_validate(
    estimator=model_nb,
    X=X_new,
    y=y_number,
    cv=10,
    scoring=type_metrics,
    return_train_score=False
)
    
    # Transformar resultados em DataFrame
    results_df = pd.DataFrame(results)

    # Calcular média e desvio padrão de cada métrica
    summary = results_df.mean().to_frame('mean')
    summary['std'] = results_df.std()
    print(f"{file_names[file]}\n")
    print(summary)