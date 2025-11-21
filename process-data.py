import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# 1. CAMINHO DO PROJETO
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)))
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

print("📂 Diretório do projeto:", PROJECT_ROOT)
print("📁 Pasta de dados:", DATA_DIR)

# arquivos
path_pacientes  = os.path.join(DATA_DIR, "dados_pacientes.csv")
path_consultas  = os.path.join(DATA_DIR, "dados_consultas.csv")
path_medicos    = os.path.join(DATA_DIR, "dados_medicos.csv")

print("\nCarregando arquivos...")

if not os.path.exists(path_pacientes):
    raise FileNotFoundError(f"❌ Arquivo não encontrado:\n{path_pacientes}")

if not os.path.exists(path_consultas):
    raise FileNotFoundError(f"❌ Arquivo não encontrado:\n{path_consultas}")

if not os.path.exists(path_medicos):
    raise FileNotFoundError(f"❌ Arquivo não encontrado:\n{path_medicos}")

# 2. CARREGAR DADOS
df_pacientes = pd.read_csv(path_pacientes)
df_consultas = pd.read_csv(path_consultas)
df_medicos   = pd.read_csv(path_medicos)

print("✔ Arquivos carregados!")
print(f"👥 Pacientes:  {len(df_pacientes)}")
print(f"📄 Consultas:  {len(df_consultas)}")
print(f"🧑‍⚕️ Médicos:  {len(df_medicos)}")

# 3. INTEGRAR DADOS
print("\n🔄 Integrando datasets...")

df = df_consultas.merge(df_pacientes, on="id_paciente", how="left")
df = df.merge(df_medicos, on="id_medico", how="left")

print("✔ Integração concluída!")
print(f"📊 Registros totais: {len(df)}")

# 4. SALVAR DATASET FINAL COMPLETO
dataset_final_path = os.path.join(DATA_DIR, "dataset_final.csv")
df.to_csv(dataset_final_path, index=False)

print("\n💾 dataset_final.csv salvo!")
print(dataset_final_path)

# 5. RFM
print("\n📊 Calculando RFM...")

df_consultas["data_consulta"] = pd.to_datetime(df_consultas["data_consulta"])
ref_date = datetime.now()

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

print("✔ RFM calculado!")


# 6. CLUSTER + RANDOM FOREST
print("\n🤖 Machine Learning...")

X = rfm[["recencia_dias", "frequencia_consultas", "valor_monetario"]]
y = rfm["frequencia_consultas"]

kmeans = KMeans(n_clusters=4, random_state=42)
rfm["cluster_rfm"] = kmeans.fit_predict(X)

model = RandomForestRegressor(random_state=42)
model.fit(X, y)

rfm["frequencia_prevista_reg"] = model.predict(X)

# Inclui informações do paciente
rfm = rfm.merge(df_pacientes[['id_paciente', 'nome', 'data_nascimento', 'sexo', 'plano_saude', 'cidade', 'possui_doenca_cronica', 'data_cadastro']],
                on='id_paciente', how='left')

# Salvar arquivo final com todas as colunas necessárias
final_path = os.path.join(DATA_DIR, "pacientes_engajamento_score.csv")
rfm.to_csv(final_path, index=False)
print("\n✔ Arquivo final salvo!")
print(final_path)
print("📈 Pacientes processados:", len(rfm))

# 7. SALVAR ARQUIVO FINAL DO APP
final_path = os.path.join(DATA_DIR, "pacientes_engajamento_score.csv")
rfm.to_csv(final_path, index=False)

print("\n✔ Arquivo final salvo!")
print(final_path)
print("📈 Pacientes processados:", len(rfm))
