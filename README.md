📘 Análise de Fatores de Conclusão e Evasão no Ensino Superior
Este repositório contém um pipeline completo de Análise Exploratória de Dados (EDA) e Modelagem Preditiva aplicado à base DM_ALUNO do Censo da Educação Superior (MEC/INEP).

O objetivo central é identificar quais fatores mais influenciam a conclusão ou a evasão de alunos no ensino superior, analisando padrões e construindo modelos de aprendizado de máquina para previsão.

⚙️ Tecnologias Utilizadas
Python 3.10+

Polars – Leitura rápida e eficiente de grandes arquivos CSV

Pandas – Manipulação de dados

Matplotlib / Seaborn – Visualização de dados

Scikit-learn – Algoritmos de Machine Learning clássicos

XGBoost – Algoritmo de Gradient Boosting

📁 Estrutura do Repositório
Plaintext

/
├── main.py                # Pipeline completo
├── README.md              # Documentação
└── resultados_trabalho/   # Criado automaticamente após execução
    ├── sample_alunos.csv
    ├── resultados_modelos.csv
    ├── feature_importance_rf.csv
    ├── feature_importance_xgb.csv
    └── *.png              # (gráficos de EDA e modelos)
Observação: O arquivo original DM_ALUNO.CSV não está incluído devido ao tamanho e possíveis restrições de distribuição.

🚀 Como Executar
1️⃣ Criar e ativar um ambiente virtual (exemplo com Conda)
Bash

conda create -n evasao python=3.10 -y
conda activate evasao
2️⃣ Instalar dependências
Bash

pip install polars pandas seaborn matplotlib scikit-learn xgboost
3️⃣ Editar o caminho do CSV no arquivo main.py
Abra o arquivo main.py e edite a variável de caminho no início do script:

Python

CSV_PATH = r"C:/SEU/CAMINHO/DM_ALUNO.CSV"
4️⃣ Executar o pipeline
Bash

python main.py
5️⃣ Resultados gerados
Os resultados serão salvos na pasta resultados_trabalho/, incluindo:

Gráficos de EDA (sexo, raça, turno, apoios, etc.)

Gráficos de distribuição

Matriz de correlação

Comparação de modelos

Métricas de classificação

Importância das variáveis

Arquivos CSV com resultados tabulados

📊 O que o pipeline faz?
[x] Lê a base usando Polars para alta performance.

[x] Seleciona variáveis de interesse relevantes para o estudo.

[x] Cria a variável alvo: Conclusão (1) vs. Evasão (0).

[x] Converte variáveis para formatos legíveis (sexo, raça, apoios, financiamento, etc.).

[x] Realiza EDA completa, incluindo:

Histogramas

Gráficos de barras

Boxplots

Heatmap de correlação

Análise univariada e bivariada

[x] Constrói um pipeline de Machine Learning com os modelos:

Regressão Logística

Random Forest

SVM (Support Vector Machine)

KNN (K-Nearest Neighbors)

XGBoost

[x] Compara todos os modelos utilizando:

Acurácia

F1-score

Matriz de confusão

[x] Exporta métricas e gráficos prontos para relatório, artigo científico ou apresentação.

📄 Licença
Este projeto é distribuído sob a licença MIT.
