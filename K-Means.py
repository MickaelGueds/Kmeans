from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist
import seaborn as sns
import geopandas as gpd
import matplotlib.patches as mpatches

df_CadUnico = pd.read_csv('visdata3-download-24-09-2024 074315.csv',encoding='ISO-8859-1')

df_CadUnico.rename(columns={'Quantidade de famílias em situação de pobreza, segundo a faixa do Programa Bolsa Família*, inscritas no Cadastro Único': 'familias_pobreza',
    'Quantidade de famílias com renda per capita mensal até meio salário-mínimo (Pobreza + Baixa renda) inscritas no Cadastro Único': 'familias_renda_meio_salario',
    'Quantidade de famílias com renda per capita mensal acima de meio salário-mínimo*** inscritas no Cadastro Único': 'familias_acima_meio_salario'}, inplace=True)

df_CadUnico_Dezembro = df_CadUnico[df_CadUnico['Referência'] == '12/2023']
print(df_CadUnico_Dezembro)

colunas_normalização = ['familias_pobreza','familias_renda_meio_salario','familias_acima_meio_salario']
dados_normalização = df_CadUnico_Dezembro[colunas_normalização]

df_populacao = pd.read_csv('tabela4709.csv')
df_populacao.rename(columns={'Tabela 4709 - População residente, Variação absoluta de população residente e Taxa de crescimento geométrico': 'Unidade Territorial'},inplace=True)
df_populacao.rename(columns={'Unnamed: 1': 'Populacao'},inplace=True)
df_populacao.drop(index=2,inplace=True)
df_populacao.drop(index=1,inplace=True)
df_populacao.drop(index=0,inplace=True)
df_populacao.drop(index=227,inplace=True)

df_populacao['Unidade Territorial'] = df_populacao['Unidade Territorial'].str.replace(' \(PI\)', '', regex=True)
df_populacao['Unidade Territorial'] = df_populacao['Unidade Territorial'].str.upper()
df_populacao.to_csv('População.csv',index=False)
print(df_populacao.columns)
print(df_populacao)


df_merge = pd.merge(df_CadUnico_Dezembro,df_populacao,on='Unidade Territorial')
df_merge['Populacao'] = pd.to_numeric(df_merge['Populacao'], errors='coerce')
colunas_variaveis = ['familias_pobreza','familias_renda_meio_salario','familias_acima_meio_salario']
for coluna in colunas_variaveis:
    df_merge[coluna] = pd.to_numeric(df_merge[coluna], errors='coerce')
    df_merge[coluna] = (df_merge[coluna]/ df_merge['Populacao']) * 100
    
print(df_merge)


scaler = StandardScaler()

dados_normalizados = scaler.fit_transform(df_merge[colunas_variaveis])
dados_normalizados_df = pd.DataFrame(dados_normalizados, columns=colunas_normalização)

print(dados_normalizados_df)

##Teste do cotovelo{
# wcss = []

# for k in range(1,11): 
#     kmeans = KMeans(n_clusters=k,random_state=0)
#     kmeans.fit(dados_normalizados)
#     wcss.append(kmeans.inertia_)


# plt.plot(range(1, 11), wcss)
# plt.title('Método do Cotovelo')
# plt.xlabel('Número de Clusters')
# plt.ylabel('WCSS')
# plt.show()

###MELHOR PONTO É O PONTO DE CLUSTERS 4
#}

kmeans = KMeans(n_clusters=4,random_state=0)
df_merge['Cluster'] = kmeans.fit_predict(dados_normalizados)
print(df_merge)

df_ordenado = df_merge[['Unidade Territorial', 'Cluster']].sort_values(by='Cluster', ascending=False)
print(df_ordenado)

# Continuando a partir do ponto onde você já tem os clusters
kmeans = KMeans(n_clusters=4,random_state=0)
df_merge['Cluster'] = kmeans.fit_predict(dados_normalizados)

# Calcular a distância de cada ponto ao centroide de seu respectivo cluster
# `cdist` calcula a distância euclidiana de cada ponto aos centroides
df_merge['Distancia_Centroide'] = cdist(dados_normalizados, kmeans.cluster_centers_[kmeans.labels_], metric='euclidean').diagonal()

# Ordenar por cluster e, dentro de cada cluster, pela distância ao centroide
df_ordenado = df_merge[['Unidade Territorial', 'Cluster', 'Distancia_Centroide']].sort_values(by=['Cluster', 'Distancia_Centroide'], ascending=[True, True])

# Exibir a tabela ordenada com as cidades mais próximas de seus respectivos centroides
print(df_ordenado)

# Se quiser salvar o resultado em um arquivo CSV para análise futura
df_ordenado.to_csv('Cidades_Ordenadas_Por_Cluster.csv', index=False)

# Plotar os clusters com a distância dos pontos
# plt.scatter(dados_normalizados[:, 0], dados_normalizados[:, 1], c=df_merge['Cluster'], cmap='viridis')
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')  # Centroides
# plt.title('Visualização dos Clusters e Centroides')
# plt.xlabel('Componente Principal 1')
# plt.ylabel('Componente Principal 2')
# plt.show()

shapefile_path = 'PI_Municipios_2022.shp'  # O caminho do arquivo .shp do Piauí
gdf = gpd.read_file(shapefile_path)

# Verifique as colunas do shapefile para garantir que tem a coluna com o nome das cidades
print(gdf.columns)

# Ajuste o merge usando a coluna 'NM_MUN' no shapefile
gdf['NM_MUN'] = gdf['NM_MUN'].str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_merge['Unidade Territorial'] = df_merge['Unidade Territorial'].str.upper().str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

gdf = gdf.merge(df_merge, left_on='NM_MUN', right_on='Unidade Territorial')

cluster_colors = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'purple'}

# Criar uma nova coluna de cores no GeoDataFrame com base nos clusters
gdf['Cluster_color'] = gdf['Cluster'].map(cluster_colors)

# Plotar o mapa com cores discretas
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Plotar o GeoDataFrame usando as cores específicas
gdf.plot(color=gdf['Cluster_color'], linewidth=0.8, ax=ax, edgecolor='0.8')

# Criar a legenda manualmente
blue_patch = mpatches.Patch(color='blue', label='Cluster 0: Regiões mais vulneráveis, com alta concentração de famílias em pobreza.')
green_patch = mpatches.Patch(color='green', label='Cluster 1: Menos pobreza, melhor situação econômica em geral, ainda com famílias de baixa renda.')
yellow_patch = mpatches.Patch(color='yellow', label='Cluster 2: Cluster mais desigual, com tanto pobreza quanto famílias com rendas maiores.')
purple_patch = mpatches.Patch(color='purple', label='Cluster 3: Níveis intermediários de pobreza e baixa renda.')

# Adicionar a legenda ao gráfico
plt.legend(handles=[blue_patch, green_patch, yellow_patch, purple_patch], loc='upper right', fontsize='small')

# Título
plt.title('Mapa do Piauí com Clusters de Cidades', fontsize=15)

# Exibir o gráfico
plt.show()

