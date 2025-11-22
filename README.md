# **Projeto Clinical Insights ML**

## **1. Visão Geral**
O projeto **Clinical Insights ML** é uma simulação completa de uma clínica médica, desenvolvida **100% em Python**, que integra:

- **Geração de dados sintéticos** (pacientes, consultas e médicos)  
- **Engenharia de features RFM** (Recência, Frequência, Valor Monetário)  
- **Modelos de Machine Learning** (Random Forest Regressor + K-Means)  
- **Dashboard interativo em Streamlit**, com gráficos interativos e métricas acionáveis  

**Objetivo:** Evitar evasão de clientes (churn) e gerar insights estratégicos para **maximizar receita e retenção**.

---

## **2. Arquitetura e Fluxo de Dados (Pipeline)**
O pipeline segue uma abordagem **simples de MLOps**:

| Etapa        | Script / Arquivo                  | Função |
|--------------|----------------------------------|--------|
| **Geração**  | `generate-df.py`                 | Cria a base de dados relacional: Pacientes, Consultas, Médicos |
| **Processamento e treinamento** | `process-data.py`         | Engenharia de features, transformação de dados e treinamento dos modelos |
| **Dados processados**  | `pacientes_engajamento_score.csv` | Dados finais consumidos pelo **dashboard Streamlit** |

---

## **3. Dados Simulados e Engenharia de Features**
**Detalhes do dataset:**

| Métrica                    | Valor / Tecnologia |
|-----------------------------|------------------|
| Volume de Pacientes         | ≈ 47.295         |
| Volume de Consultas         | ≈ 97.083         |
| Geração de Dados            | Python (Faker + Random) com `seed=42` para reprodutibilidade |
| Data Base de Referência     | 2025-05-20 (usada para calcular `recencia_dias`) |

**Features RFM criadas:**

- **Recência (`recencia_dias`)** – Dias desde a última consulta (principal fator de risco de churn com 98,5%)  
- **Frequência (`frequencia_consultas`)** – Total de consultas por paciente  
- **Valor Monetário (`valor_monetario`)** – Total gasto na clínica  

**Transformação aplicada:**  
- **Logaritmo natural com offset**: `np.log1p(x)` para estabilizar valores dispersos antes da clusterização (tratamento de outliers).

---

## **4. Modelos de Machine Learning**

| Modelo                  | Tipo                | Função                                                                 | Performance |
|-------------------------|------------------|------------------------------------------------------------------------|-------------|
| **K-Means**             | Não-Supervisionado | Clusteriza pacientes em perfis RFM (ex.: "VIPS", "EM EVASÃO")          | Estável após log transform |
| **Random Forest Regressor** | Supervisionado    | Calcula **Score de Engajamento** (`frequencia_prevista_reg`)            | R² ≈ 0.472 (47% da variação explicada) |

**Insight principal:**  

- **Recência (`recencia_dias`)** → **98,5% do poder preditivo** do modelo para estimar a frequência futura  
- **Conclusão estratégica:** A clínica deve **priorizar pacientes com alta recência, alto valor monetário e baixa frequência**, usando KPIs específicos para evitar churn e maximizar receita.

---

## **5. Dashboard Interativo (Streamlit)**

**Tecnologias usadas:**

| Tecnologia | Função |
|------------|--------|
| Streamlit  | Criar dashboard web interativo, deploy em cloud |
| Plotly     | Gráficos interativos: scatter plots e barras |

**Recursos do dashboard:**

- **Clusters RFM visualizados graficamente**  
- **Escala logarítmica** no eixo Y para valores monetários discrepantes (R$ 0 a R$ 6.000)
- **Score de Engajamento** baseado na média de frequência anual
- **Tabela de priorização de pacientes:** Top 10 por cluster (recência + valor monetário), destacando **pacientes de maior risco financeiro**  

**Acesse o dashboard clicando [aqui](https://dashboard-clinica-medica.streamlit.app/)**


---

## **6. Conclusão e Impacto**
O projeto **Clinical Insights ML** demonstra a **aplicabilidade prática de Python** em todo o fluxo de dados, do **tratamento à visualização interativa**:

- Segmentação de clientes para **estratégias de retenção**  
- Previsão de comportamento financeiro com **Random Forest**  
- Ação direta baseada em insights RFM para **maximizar receita**  

**Principais aprendizados:**

- Engenharia de features RFM é altamente eficaz para **prever churn**  
- A recência é a métrica mais crítica para **tomada de decisão clínica e financeira**  
- Integração de **ML + dashboard** fornece **insights acionáveis** em tempo real  

**Tecnologias usadas:** Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit.
