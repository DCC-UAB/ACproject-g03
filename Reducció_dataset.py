# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:43:31 2024

@author: 04gao
"""

import pandas as pd

def process_and_save_dataset(file_path, output_path, sample_size=500000):
    """
    Processa un dataset CSV seleccionant 500.000 files i columnes específiques,
    i guarda el resultat en un nou fitxer CSV.

    Args:
        file_path (str): El camí al fitxer CSV original del dataset.
        output_path (str): El camí on es guardarà el nou fitxer CSV processat.
        sample_size (int): El nombre de files a seleccionar aleatòriament (per defecte 500.000).
        
    Returns:
        None
    """
    try:
        # Llegeix el dataset complet
        print("Carregant el dataset...")
        dataset = pd.read_csv(file_path)

        # Comprova si les columnes necessàries hi són
        required_columns = ['Id', 'review/score', 'review/text']
        if not all(column in dataset.columns for column in required_columns):
            raise ValueError(f"Les columnes requerides no estan al dataset. Calen {required_columns}")
        
        # Mostra informació del dataset original
        print(f"Dataset carregat amb {len(dataset)} files i {len(dataset.columns)} columnes.")
        
        # Selecciona només les columnes necessàries
        dataset_reduced = dataset[required_columns]
        
        # Selecciona un sample aleatori de les files
        print(f"Seleccionant {sample_size} files aleatòries...")
        sampled_dataset = dataset_reduced.sample(n=sample_size, random_state=42)
        
        # Guarda el resultat en un nou fitxer CSV
        print(f"Guardant el nou dataset a {output_path}...")
        sampled_dataset.to_csv(output_path, index=False)
        
        print("El processament s'ha completat correctament. Dataset guardat amb èxit.")

    except Exception as e:
        print(f"Hi ha hagut un error: {e}")

process_and_save_dataset("Books_rating.csv", "reduced_reviews.csv")

