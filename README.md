# 📘 Análise de Fatores de Conclusão e Evasão no Ensino Superior

Este repositório contém um pipeline completo de **Análise Exploratória de Dados (EDA)** e **Modelagem Preditiva** aplicado à base `DM_ALUNO` do Censo da Educação Superior (MEC/INEP).

O objetivo central é identificar **quais fatores mais influenciam a conclusão ou a evasão** de alunos no ensino superior, analisando padrões e construindo modelos de aprendizado de máquina para previsão.

## ⚙️ Tecnologias Utilizadas
- Python 3.10+
- Polars
- Pandas
- Matplotlib / Seaborn
- Scikit-learn
- XGBoost

## 📁 Estrutura do Repositório
/
├── main.py
├── README.md
└── resultados_trabalho/
      ├── sample_alunos.csv
      ├── resultados_modelos.csv
      ├── feature_importance_rf.csv
      ├── feature_importance_xgb.csv
      ├── *.png

## 🚀 Como Executar
1. Criar ambiente virtual:
   conda create -n evasao python=3.10 -y
   conda activate evasao

2. Instalar dependências:
   pip install polars pandas seaborn matplotlib scikit-learn xgboost

3. Editar caminho do CSV em main.py:
   CSV_PATH = r"C:/SEU/CAMINHO/DM_ALUNO.CSV"

4. Executar:
   python main.py

## 📊 Resultados
Gerados automaticamente em:
resultados_trabalho/

## 📄 Licença
MIT License.
