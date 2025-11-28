import pandas as pd
from faker import Faker
import random
import os
# %%

base_dir = os.path.dirname(__file__)
pacientes_path = os.path.join(base_dir, '..', '..', 'data', 'dados_pacientes.csv')
pacientes = pd.read_csv(pacientes_path)

# %%
Faker.seed(42)
random.seed(42)
fake = Faker('pt-BR')

num_consultas = 97083
num_pacientes_total = pacientes['id_paciente'].max()

def get_valor_consulta(plano):
    if plano == 'Popular':
        return 0
    elif plano == 'Executivo':
        return 100
    elif plano == 'Premium':
        return 500
    
consultas = []
for i in range(1, num_consultas + 1):
    paciente_id = random.choice(range(1, num_pacientes_total + 1)) 
    medico_id = random.choice(range(1,501))
    data_consulta = fake.date_between(start_date='-1y', end_date='today')
    
    # Busca o plano de saúde do paciente selecionado
    try:
        plano = pacientes[pacientes['id_paciente'] == paciente_id]['plano_saude'].values[0]
        valor_consulta = get_valor_consulta(plano)
    except IndexError:
        plano = 'Popular'
        valor_consulta = 0
        
    consultas.append([
        paciente_id, 
        medico_id, 
        data_consulta,
        valor_consulta
        ])
    
df_consultas = pd.DataFrame(consultas, columns=[
    'id_paciente',
    'id_medico',
    'data_consulta',
    'valor_consulta'
    ])
# %%

output_path = os.path.join(base_dir, '..', '..', 'data', 'dados_consultas.csv')
df_consultas.to_csv(output_path, index=False, encoding='utf-8', errors='replace')

# %%

import pandas as pd
from faker import Faker
import random
import os
# %%
Faker.seed(42)
random.seed(42)
fake = Faker('pt-BR')

num_medicos = 500
especialidades = ['Cardiologista', 'Pediatra', 'Oftalmologista', 'Dermatologista', 'Ortopedista', 'Ginecologista', 'Urologista']
sexos = ['M', 'F']

dados_medicos = []

for i in range(1, num_medicos + 1):
    sexo = random.choice(sexos)
    if sexo == 'M':
        nome = fake.name_male()
        titulo = 'Dr.'
    else:
        nome = fake.name_female()
        titulo = 'Dra.'
    if 'Dr.' not in nome and 'Dra.' not in nome:
        nome = f"{titulo} {nome}"
    especialidade = random.choice(especialidades)
    crm = f"CRM{random.randint(100000, 999999)}"
    cidade = fake.city()
    telefone = fake.phone_number()
    dados_medicos.append([
        i,
        nome,
        sexo,
        especialidade,
        crm,
        cidade,
        telefone
        ])

df_medicos = pd.DataFrame(dados_medicos, columns=[
    'id_medico',
    'nome',
    'sexo',
    'especialidade',
    'crm',
    'cidade',
    'telefone'
    ])
# %% 

base_dir = os.path.dirname(__file__)
output_path = os.path.join(base_dir, '..', '..', 'data', 'dados_medicos.csv')

df_medicos.to_csv(output_path, index=False, encoding='utf-8', errors='replace')

# %%

import pandas as pd
from faker import Faker
import random
import os
# %%
Faker.seed(42)
random.seed(42)
fake = Faker('pt-BR')
num_pacientes = 47295 
planos = ['Popular', 'Executivo', 'Premium']
sexos = ['M', 'F']

dados_pacientes = []

for i in range(1, num_pacientes + 1):
    sexo = random.choice(sexos)
    nome = fake.name_male() if sexo == 'M' else fake.name_female()
    data_nascimento = fake.date_of_birth(minimum_age=8, maximum_age=90)
    cidade = fake.city()
    plano = random.choices(planos, weights=[0.7, 0.2, 0.1])[0]
    possui_doenca_cronica = random.choices([True, False], weights=[0.15, 0.85])[0]
    data_cadastro = fake.date_between(start_date='-15y', end_date='today')
    dados_pacientes.append([
        i,
        nome,
        sexo,
        data_nascimento,
        cidade,
        plano,
        possui_doenca_cronica,
        data_cadastro
        ])
    
df_pacientes = pd.DataFrame(dados_pacientes, columns=[
    'id_paciente',
    'nome',
    'sexo',
    'data_nascimento',
    'cidade',
    'plano_saude',
    'possui_doenca_cronica',
    'data_cadastro'
    ])
# %% 

base_dir = os.path.dirname(__file__)
output_path = os.path.join(base_dir, '..', '..', 'data', 'dados_pacientes.csv')

df_pacientes.to_csv(output_path, index=False, encoding='utf-8', errors='replace')















