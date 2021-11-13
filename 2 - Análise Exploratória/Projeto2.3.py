# Imports
import os
import subprocess
import stat
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mat
import matplotlib.pyplot as plt
from datetime import datetime
sns.set(style="white")

# Dataset
clean_data_path = "dataset/autos.csv"
df = pd.read_csv(clean_data_path,encoding = "latin-1")

#Preço médio do veículo por tipo de combustível e tipo de caixa de câmbio
sns.set_style("whitegrid")
sns.barplot(y='price', x="fuelType", data=df, hue="gearbox")
plt.title('Preço médio dos veículos por Combustível')
plt.ylabel("Preço médio de Veículos")
plt.xlabel("Tipo de Combustível")
plt.show()

#Potência média de um veículo por tipo de veículo e tipo de caixa de câmbio
sns.set_style("whitegrid")
sns.barplot(y='powerPS', x="vehicleType", data=df, hue="gearbox")
plt.title('Potência média de um veículo')
plt.ylabel("Potencia média")
plt.xlabel("Tipo de Veículo")
plt.show()
