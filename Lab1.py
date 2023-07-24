# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:09:56 2023

@author: juana
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline

df = pd.read_csv("risk_factors_cervical_cancer.csv")

#6
df.isna().sum()
df.replace("?", np.NAN, inplace=True)

df.isna().sum()/len(df)*100

df[df["Smokes"].isna()].isna().sum()
df_drop_Smokes = df.dropna(subset=["Smokes"])
df_drop_Smokes_S = df_drop_Smokes.drop(columns=df.columns[[26, 27]])
df_drop_Smokes_S.isna().sum()
df_drop_small_na = df_drop_Smokes_S.dropna(subset=["Num of pregnancies", "First sexual intercourse","Number of sexual partners"])
df_drop_small_na.isna().sum() / len(df_drop_small_na) * 100


imp = [7, 9, 11]
imp2 = (13,24)
imp3 = (26,33)
#imp = [7, 9, 11, (13,24), (26,33)]
def fix_nulls_in_dataframe(df, columns_indices):
    def fix_column_nulls(column):
        # Calculamos las proporciones de 0's y 1's en la columna
        value_counts = column.value_counts(normalize=True)
        prop_0 = value_counts[0] if 0 in value_counts else 0
        prop_1 = 1 - prop_0
        
        # Generamos una máscara para los valores nulos
        mask_nulls = column.isnull()
        
        # Generamos valores aleatorios en base a las proporciones y reemplazamos los nulos
        random_values = np.random.choice([0, 1], size=mask_nulls.sum(), p=[prop_0, prop_1])
        column[mask_nulls] = random_values
    
    if isinstance(columns_indices, int):
        columns_indices = [columns_indices]
    elif isinstance(columns_indices, tuple):
        columns_indices = list(range(columns_indices[0], columns_indices[1]+1))
    
    for index in columns_indices:
        column_name = df.columns[index]
        fix_column_nulls(df[column_name])
    
    return df


# Aplicar la imputación a las columnas seleccionadas
fix_nulls_in_dataframe(df_drop_small_na, imp)
fix_nulls_in_dataframe(df_drop_small_na, imp2)
fix_nulls_in_dataframe(df_drop_small_na, imp3)

knn_imputer = KNNImputer(n_neighbors=5)
df_drop_small_na[["Hormonal Contraceptives (years)", "IUD (years)", "STDs (number)"]] = knn_imputer.fit_transform(df_drop_small_na[["Hormonal Contraceptives (years)", "IUD (years)", "STDs (number)"]])



booleanos = [4, 7, 9, 11, (13,25), (26,34)]
for col in booleanos:
    if isinstance(col, int):
        df_drop_small_na.iloc[:, col] = df_drop_small_na.iloc[:, col].astype(float)
    elif isinstance(col, tuple):
        start, end = col
        df_drop_small_na.iloc[:, start:end] = df_drop_small_na.iloc[:, start:end].astype(float)
        
df_drop_small_na.to_csv("DataLimpia.csv")
data = df_drop_small_na
#PCA
for col in [1,2,3,5,6]:
    data.iloc[:,col] = data.iloc[:,col].astype(float)

chi_cuadrado, p_valor = calculate_bartlett_sphericity(data.iloc[:,0:33])
chi_cuadrado, p_valor
#se rechaza la H nula
kmo,kmo_modelo = calculate_kmo(data)
kmo_modelo

pca_pipe = make_pipeline(StandardScaler(),PCA()) #Se escalan los datos y luego se le aplica PCA
pca_pipe.fit(data)
#Se extrae el modelo del pipeline
modelo_pca = pca_pipe.named_steps['pca']

#Se convierte el arreglo en data frame
pd.DataFrame(
    data = modelo_pca.components_,
    columns = data.columns,
    #index = ['PC1','PC2','PC3','PC5','PC6','PC7','PC8']
)

fig, ax = plt.subplots(nrows=1, ncols=1)
componentes = modelo_pca.components_
plt.imshow(componentes.T)
plt.yticks(range(len(data.columns)), data.columns)
plt.xticks(range(len(data.columns)), np.arange(modelo_pca.n_components_)+1, )
plt.grid(False)
plt.colorbar();


print('----------------------------------------------------')
print('Porcentaje de varianza explicada por cada componente')
print('----------------------------------------------------')
print(modelo_pca.explained_variance_ratio_)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.bar(
    x      = np.arange(modelo_pca.n_components_) + 1,
    height = modelo_pca.explained_variance_ratio_
)

for x, y in zip(np.arange(len(data.columns)) + 1, modelo_pca.explained_variance_ratio_):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )

ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_ylim(0, 1.1)
ax.set_title('Porcentaje de varianza explicada por cada componente')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza explicada')


# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
ax.plot(
    np.arange(len(data.columns)) + 1,
    prop_varianza_acum,
    marker = 'o'
)

for x, y in zip(np.arange(len(data.columns)) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada')









