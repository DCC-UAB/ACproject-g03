import pandas as pd
import spacy

# Carrega el model spaCy
nlp = spacy.load('en_core_web_sm')

# # Divideix la lematització en chunks
# def process_chunk(chunk, batch_size=50):
#     """Processa un chunk de dades amb lematització."""
#     texts = chunk['review'].tolist()
#     docs = nlp.pipe(texts, batch_size=batch_size, disable=['parser', 'ner'])
#     chunk['lemmatized_review'] = [
#         " ".join([token.lemma_ for token in doc if not token.is_stop]) for doc in docs
#     ]
#     return chunk[['Id', 'score', 'lemmatized_review']]  # Manté les columnes necessàries

# # Defineix la mida dels chunks
# chunk_size = 10_000  # Processarem 10.000 files a la vegada
# output_file = 'lemmatized_reviews_only.csv'

# # Processa i desa per chunks
# for i, chunk in enumerate(pd.read_csv('reduced_reviews.csv', chunksize=chunk_size)):
#     # Renombra les columnes per facilitar el processament
#     chunk.rename(columns={'review/score': 'score', 'review/text': 'review'}, inplace=True)
    
#     # Comprova que la columna 'ID' existeix
#     if 'Id' not in chunk.columns:
#         raise ValueError("El dataset original no conté una columna 'ID'.")
    
#     # Processa el chunk
#     processed_chunk = process_chunk(chunk)
    
#     # Desa cada chunk al fitxer de sortida
#     if i == 0:
#         processed_chunk.to_csv(output_file, index=False, mode='w')  # Escriu encapçalament
#     else:
#         processed_chunk.to_csv(output_file, index=False, mode='a', header=False)  # Sense encapçalament
    
#     print(f"Chunk {i + 1} processat i desat.")

# print("Processament complet.")


# REVISIO MANUAL EXEMPLES PROCESSATS i STOPWORDS ELIMINADES

# Carrega els datasets
original_df = pd.read_csv("reduced_reviews.csv")
lemmatized_df = pd.read_csv("lemmatized_reviews_only.csv")

# Comprova que els datasets tinguin el mateix índex
if not original_df.index.equals(lemmatized_df.index):
    raise ValueError("Els índexs dels datasets original i lematitzat no coincideixen.")

# Agafa un exemple per comparar
sample_index = 7  # Pots canviar aquest índex
original_text = original_df.loc[sample_index, "review/text"]
lemmatized_text = lemmatized_df.loc[sample_index, "lemmatized_review"]

# Analitza el text original amb spaCy
doc = nlp(original_text)

# Paraules eliminades (stopwords)
removed_stopwords = [token.text for token in doc if token.is_stop]

# Verbs lematitzats
lemmatized_verbs = [(token.text, token.lemma_) for token in doc if token.pos_ == "VERB"]

# Resultats
print(f"Text original:\n{original_text}\n")
print(f"Text lematitzat:\n{lemmatized_text}\n")
print(f"Stopwords eliminades:\n{removed_stopwords}\n")
print(f"Verbs lematitzats (original -> lematitzat):\n{lemmatized_verbs}\n")