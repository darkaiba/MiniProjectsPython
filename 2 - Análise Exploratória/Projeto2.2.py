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

#Número de veículos pertencentes a cada marca
sns.set_style("whitegrid")
sns.catplot(y="brand", data=df, kind="count")
plt.title('Número de veículos pertencentes a cada marca')
plt.xlabel("Numero de Veículos")
plt.ylabel("Marcas")
plt.show()

#Preço médio dos veículos com base no tipo de veículo, bem como no tipo de caixa de câmbio
sns.set_style("whitegrid")
sns.barplot(x='price', y="vehicleType", data=df, hue="gearbox")
plt.title('Preço médio dos veículos')
plt.xlabel("Preço médio de Veículos")
plt.ylabel("Tipo Veículos")
plt.show()


