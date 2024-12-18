import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(file_path)

def check_balance(data, column):
    """Verifica si la columna especificada está balanceada y genera un gráfico de distribución."""
    if column not in data.columns:
        raise ValueError(f"La columna '{column}' no está en el dataset.")

    # Contar la frecuencia de cada categoría
    class_counts = data[column].value_counts()
    total = class_counts.sum()

    print("Distribución de la columna", column)
    for category, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"{category}: {count} ({percentage:.2f}%)")

    # Visualizar la distribución
    class_counts.plot(kind='bar', title=f"Distribución de {column}", xlabel=column, ylabel="Frecuencia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(file_path, column):
    """Ejecuta el chequeo de balanceo en el dataset."""
    print("Cargando datos...")
    data = load_data(file_path)

    print("Verificando balanceo...")
    check_balance(data, column)

if __name__ == "__main__":
    # Configuración del script
    FILE_PATH = "Books_rating.csv"  # Ruta del archivo
    COLUMN = "review/score"  # Columna a verificar

    main(FILE_PATH, COLUMN)
