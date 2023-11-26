import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
from perceptron_class import Perceptron
from geneticAlg_class import GeneticAlgorithm
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from numpy import meshgrid


def train_perceptron_with_genetic_algorithm(df, test_size=0.3, perceptron_epochs=50, ga_epochs=50, population_size=50):

    path_save_graphics = "D:\\Projetos\\bio_insp_codes\\trabalho_final\\graphics\\"

    X = df[["sepal_length", "sepal_width"]].iloc[:100].values
    y = df.iloc[0:100].species.values
    y = np.where(y == 'setosa', -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Inicializa o Algoritmo Genético
    ga = GeneticAlgorithm(population_size=population_size, perceptron_epochs=perceptron_epochs)

    # Executa o Algoritmo Genético para obter os melhores pesos
    best_weights = ga.run(X_train, y_train, ga_epochs)

    # Cria e treina um perceptron com os melhores pesos encontrados
    perceptron = Perceptron(epochs=perceptron_epochs)
    perceptron.set_weights(best_weights)
    perceptron.train(X_train, y_train)

    # Faz previsões nos dados de teste
    y_test_predicted = perceptron.predict(X_test)


    # Calcula e imprime as métricas
    accuracy = accuracy_score(y_test, y_test_predicted)
    precision = precision_score(y_test, y_test_predicted)
    recall = recall_score(y_test, y_test_predicted)
    f1 = f1_score(y_test, y_test_predicted)
    conf_matrix = confusion_matrix(y_test, y_test_predicted)

    print(f"Acurácia: {accuracy * 100:.2f}%")
    print(f"Precisão: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1 * 100:.2f}%")
    print("Matriz de Confusão:")
    print(conf_matrix)

    # Calcula a acurácia
    accuracy = np.mean(y_test_predicted == y_test)
    print(f"Acurácia na previsão dos dados de teste ({test_size * 100}%): {accuracy * 100:.2f}%")
    
    # Plota as regiões de decisão do Perceptron
    plt.figure(figsize=(10, 8))

    # Adicione a criação da grade de pontos
    xx, yy = meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                      np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))

    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

    # Plota os pontos de treinamento
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')
    plt.title(f"Decision Regions Plot ({test_size * 100}% Training Data)", fontsize=18)
    plt.xlabel("sepal length [cm]", fontsize=15)
    plt.ylabel("sepal width [cm]", fontsize=15)
    plt.savefig(os.path.join(path_save_graphics, f"plot_{int(test_size * 100)}trained.png"))
    plt.close()

    # Plota o gráfico de erros do Perceptron
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(perceptron.errors_)+1), perceptron.errors_, marker="o", label="error plot")
    plt.xlabel("Epochs")
    plt.ylabel("Missclassifications")
    plt.legend()
    plt.savefig(os.path.join(path_save_graphics, f"error_plot_{int(test_size * 100)}.png"))
    plt.close()

    return perceptron, X_test, y_test, y_test_predicted
    

# --------------------- Carrega os dados --------------------- #
df = pd.read_csv("D:\\Projetos\\bio_insp_codes\\trabalho_final\\dataset\\iris_dataset.csv", sep=",")

# Proporções de dados de teste
proportions = [0.3, 0.5, 0.7]

# Treinamento do Perceptron com Algoritmo Genético para diferentes proporções
for proportion in proportions:
    trained_perceptron, X_test, y_test, y_test_predicted = train_perceptron_with_genetic_algorithm(df, test_size=proportion, perceptron_epochs=10, ga_epochs=50, population_size=25)

    print(f"Predições dos dados restantes ({(1 - proportion) * 100}%): {y_test_predicted}")
