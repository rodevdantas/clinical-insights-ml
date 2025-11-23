import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
########## Importações necessárias para as validações ######### (para quem for rodar os comentários)
# from sklearn.metrics import silhouette_score 
# import matplotlib.pyplot as plt

# 1. CAMINHO DO PROJETO
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print(" Diretório do projeto:", PROJECT_ROOT)
print(" Pasta de dados:", DATA_DIR)

# arquivos
path_pacientes  = os.path.join(DATA_DIR, "dados_pacientes.csv")
path_consultas  = os.path.join(DATA_DIR, "dados_consultas.csv")
path_medicos    = os.path.join(DATA_DIR, "dados_medicos.csv")

print("\nCarregando arquivos...")

if not os.path.exists(path_pacientes):
    raise FileNotFoundError(f" Arquivo não encontrado:\n{path_pacientes}")

if not os.path.exists(path_consultas):
    raise FileNotFoundError(f" Arquivo não encontrado:\n{path_consultas}")

if not os.path.exists(path_medicos):
    raise FileNotFoundError(f" Arquivo não encontrado:\n{path_medicos}")

# 2. CARREGAR DADOS
df_pacientes = pd.read_csv(path_pacientes)
df_consultas = pd.read_csv(path_consultas)
df_medicos   = pd.read_csv(path_medicos)

print(" Arquivos carregados!")
print(f" Pacientes:  {len(df_pacientes)}")
print(f" Consultas:  {len(df_consultas)}")
print(f" Médicos:  {len(df_medicos)}")

# 3. INTEGRAR DADOS
print("\n Integrando datasets...")

df = df_consultas.merge(df_pacientes, on="id_paciente", how="left")
df = df.merge(df_medicos, on="id_medico", how="left")

print(" Integração concluída!")
print(f" Registros totais: {len(df)}")

# 4. SALVAR DATASET FINAL COMPLETO
dataset_final_path = os.path.join(DATA_DIR, "dataset_final.csv")
df.to_csv(dataset_final_path, index=False)

print("\n dataset_final.csv salvo!")
print(dataset_final_path)

# 5. RFM
print("\n Calculando RFM...")

df_consultas["data_consulta"] = pd.to_datetime(df_consultas["data_consulta"])
ref_date = datetime(2025, 11, 22)

rfm = df_consultas.groupby("id_paciente").agg({
    "data_consulta": lambda x: (ref_date - x.max()).days,   # Recência
    "valor_consulta": "sum"                                 # Monetário
}).reset_index()

# Frequência = número de consultas (contagem de linhas)
frequencia = df_consultas.groupby("id_paciente").size().reset_index(name="frequencia_consultas")

rfm = rfm.merge(frequencia, on="id_paciente", how="left")

rfm.columns = [
    "id_paciente",
    "recencia_dias",
    "valor_monetario",
    "frequencia_consultas"
]

print(" RFM calculado!")


# 6. CLUSTER + RANDOM FOREST
print("\n Machine Learning...")

# Definindo as variáveis para o modelo RFM
X = rfm[["recencia_dias", "frequencia_consultas", "valor_monetario"]]
y = rfm["frequencia_consultas"]

# ==============================================================================
# 🌟 CÓDIGO DE VALIDAÇÃO DE CLUSTERS (Elbow Method e Silhouette Score)
#
# A análise estatística indicou K=2 como o ideal.
# No entanto, K=4 foi escolhido para o projeto para oferecer granularidade acionável
# ao negócio (VIPS, OBS. Moderada, OBS. Leve e Baixo Impacto).
# O Silhouette Score de K=4 (~0.61) é considerado robusto.
# ==============================================================================


# ------------------ MÉTODO DO COTOVELO (ELBOW METHOD) ------------------
# Descomente para rodar a validação da inércia (WCSS)
#wcss = []
#k_range = range(1, 11)
#print("\n Rodando Elbow Method (K=1 a K=10)...")
#for k in k_range:
#    if k <= len(X):
#        kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
#        kmeans_model.fit(X)
#        wcss.append(kmeans_model.inertia_)
#    else:
#        wcss.append(None)

# # Plotagem do Elbow Method
#if 'plt' in globals():
#    plt.figure(figsize=(10, 6))
#    plt.plot(k_range, wcss, marker='o', linestyle='--')
#    plt.title('Método do Cotovelo (WCSS vs K)')
#    plt.xlabel('Número de Clusters (K)') 
#    plt.ylabel('WCSS (Inertia)')
#    plt.xticks(k_range)
#    plt.grid(True)
#    plt.show()
#else:
#    print("ATENÇÃO: Importe 'matplotlib.pyplot as plt' para visualizar o gráfico do Elbow.")
# ------------------------------------------------------------------------


# ------------------ COEFICIENTE DE SILHUETA (SILHOUETTE SCORE) ------------------
# Descomente para rodar a validação da qualidade de separação dos clusters
#silhouette_scores = {}
#k_range_sil = range(2, 11)
#print("\nRodando Silhouette Score (K=2 a K=10)...")
#
#for k in k_range_sil:
#    kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
#    cluster_labels = kmeans_model.fit_predict(X)
#
#     # O cálculo é O(N^2) e pode ser lento para grandes datasets
#    score = silhouette_score(X, cluster_labels)
#    silhouette_scores[k] = score
#
#print("\n--- Resultados do Silhouette Score ---")
#for k, score in silhouette_scores.items():
#    print(f"K = {k}: Score = {score:.4f}")
# # Exemplo dos resultados para 41.284 pacientes: K=2 (0.8384), K=4 (0.6176)
#
# # Plotagem do Silhouette Score
#if 'plt' in globals():
#    plt.figure(figsize=(10, 6))
#    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o', linestyle='-')
#    plt.title('Silhouette Score vs K')
#    plt.xlabel('Número de Clusters (K)')
#    plt.ylabel('Silhouette Score Médio')
#    plt.xticks(list(silhouette_scores.keys()))
#    plt.grid(True)
#    plt.show()
# else:
#    print("ATENÇÃO: Importe 'matplotlib.pyplot as plt' para visualizar o gráfico da Silhueta.")
# ----------------------------------------------------------------------------------


# --- CLUSTERIZAÇÃO RFM FINAL (K=4) ---
# K=4 é a escolha de negócio, validada pelo Silhouette Score > 0.60
K_ESCOLHIDO = 4
print(f"\n Aplicando K-Means com K={K_ESCOLHIDO} (Escolha de Negócio)...")
kmeans = KMeans(n_clusters=K_ESCOLHIDO, random_state=42, n_init=10)
rfm["cluster_rfm"] = kmeans.fit_predict(X)


# --- TREINAMENTO DO MODELO PREDITIVO (Random Forest Regressor) ---
# Modelo para prever a Frequência de Consultas (Score de Engajamento)
print(" Treinando Random Forest Regressor para Score de Engajamento...")
model = RandomForestRegressor(random_state=42)
model.fit(X, y)
rfm["frequencia_prevista_reg"] = model.predict(X)

# Inclui informações do paciente
rfm = rfm.merge(df_pacientes[['id_paciente', 'nome', 'data_nascimento', 'sexo', 'plano_saude', 'cidade', 'possui_doenca_cronica', 'data_cadastro']],
                on='id_paciente', how='left')

# Salvar arquivo final com todas as colunas necessárias
final_path = os.path.join(DATA_DIR, "pacientes_engajamento_score.csv")
rfm.to_csv(final_path, index=False)

print("\n Arquivo final salvo!")
print(final_path)
print(" Pacientes processados:", len(rfm))