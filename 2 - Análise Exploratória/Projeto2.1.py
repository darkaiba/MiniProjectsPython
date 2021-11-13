# Imports
import os
import subprocess
import stat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
sns.set(style = "white")

# Dataset
clean_data_path = "dataset/autos.csv"
df = pd.read_csv(clean_data_path,encoding = "latin-1")

#Distribuição de Veículos com base no Ano de Registro
print(df.columns)

# Crie um Plot com a Distribuição de Veículos com base no Ano de Registro
fig, ax = plt.subplots()
sns.histplot(df["yearOfRegistration"], color="#33cc33", kde=True)
plt.title('Distribuição de Veículos com base no Ano de Registro')
plt.ylabel("Densidade (KDE)")
plt.xlabel("Ano de Registro")
plt.show()

#Variação da faixa de preço pelo tipo de veículo
sns.set_style("whitegrid")
sns.boxplot(df['vehicleType'],  df['price'], data=df)
plt.title('Análise de Outliers')
plt.ylabel("Preço")
plt.xlabel("Tipo de Veiculo")
plt.show()

#Contagem total de veículos à venda conforme o tipo de veículo
sns.set_style("whitegrid")
sns.catplot(x="vehicleType", data=df, kind="count")
plt.title('Contagem Total de Vendas dos Veículos')
plt.ylabel("Total de Vendas")
plt.xlabel("Tipo de Veiculo")
plt.show()
