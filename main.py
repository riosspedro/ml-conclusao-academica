"""
Análise de fatores de conclusão/evasão no ensino superior
=========================================================

Pipeline completo:
- Leitura da base DM_ALUNO com Polars
- Amostragem de 2% para viabilizar o processamento
- EDA (gráficos salvos em pasta)
- Treino e avaliação de modelos de classificação
- Importância de variáveis (Random Forest e XGBoost)
- Resultados salvos em CSV e gráficos

Autor: (seu nome)
"""

import os

import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------------------------------------------------


CSV_PATH = r"C:/CAMINHO/DM_ALUNO.CSV"  # <<< ajuste o caminho aqui
RESULTS_DIR = "resultados_trabalho"


sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def ensure_results_dir() -> None:
    """Cria a pasta de resultados, se não existir."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def savefig(filename: str) -> None:
    """Salva a figura atual na pasta de resultados e fecha o plot."""
    path = os.path.join(RESULTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIG] {path}")


# ----------------------------------------------------------------------
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ----------------------------------------------------------------------


def load_data(csv_path: str) -> pl.DataFrame:
    """Lê a base DM_ALUNO com Polars, mantendo apenas as colunas relevantes."""
    colunas_usadas = [
        "NU_IDADE",
        "TP_SEXO",
        "TP_COR_RACA",
        "TP_TURNO",
        "IN_INGRESSO_ENEM",
        "IN_INGRESSO_VESTIBULAR",
        "IN_FINANCIAMENTO_ESTUDANTIL",
        "IN_APOIO_ALIMENTACAO",
        "IN_APOIO_TRANSPORTE",
        "IN_APOIO_MORADIA",
        "TP_SITUACAO",
    ]

    df_pl = pl.read_csv(
        csv_path,
        separator="|",
        columns=colunas_usadas,
        ignore_errors=True,
    )

    # ALVO = 1 se TP_SITUACAO == 2 (concluiu)
    df_pl = df_pl.with_columns(
        (pl.col("TP_SITUACAO") == 2).cast(pl.Int64).alias("ALVO")
    )

    return df_pl


def sample_to_pandas(df_pl: pl.DataFrame, frac: float = 0.02) -> pd.DataFrame:
    """Amostra uma fração da base com Polars e converte para Pandas."""
    sample = df_pl.sample(fraction=frac, seed=42)
    sample_path = os.path.join(RESULTS_DIR, "sample_alunos.csv")
    sample.write_csv(sample_path)
    print(f"[DATA] Amostra salva em {sample_path}")

    dfp = pd.read_csv(sample_path)
    return dfp


def map_categorical_columns(dfp: pd.DataFrame) -> pd.DataFrame:
    """Mapeia variáveis categóricas para rótulos legíveis e cria colunas auxiliares."""

    sexo_map = {1: "Masculino", 2: "Feminino"}
    dfp["TP_SEXO"] = dfp["TP_SEXO"].map(sexo_map)

    raca_map = {
        0: "Não declarado",
        1: "Branca",
        2: "Preta",
        3: "Parda",
        4: "Amarela",
        5: "Indígena",
        9: "Não informada",
    }
    dfp["TP_COR_RACA"] = dfp["TP_COR_RACA"].map(raca_map).fillna("Não informado")

    turno_map = {1: "Matutino", 2: "Vespertino", 3: "Noturno", 4: "Integral"}
    dfp["TP_TURNO"] = dfp["TP_TURNO"].map(turno_map).fillna("Outro/Não informado")

    def tipo_ingresso(row):
        if row.get("IN_INGRESSO_ENEM", 0) == 1:
            return "ENEM"
        if row.get("IN_INGRESSO_VESTIBULAR", 0) == 1:
            return "Vestibular"
        return "Outros"

    dfp["TIPO_INGRESSO"] = dfp.apply(tipo_ingresso, axis=1)

    # alvo numérico
    dfp["ALVO"] = pd.to_numeric(dfp["ALVO"], errors="coerce").fillna(0).astype(int)

    # labels binários para EDA
    bin_label_map = {0: "Não", 1: "Sim"}
    dfp["FINANCIAMENTO_LABEL"] = dfp["IN_FINANCIAMENTO_ESTUDANTIL"].map(bin_label_map)
    dfp["APOIO_ALIM_LABEL"] = dfp["IN_APOIO_ALIMENTACAO"].map(bin_label_map)
    dfp["APOIO_TRANS_LABEL"] = dfp["IN_APOIO_TRANSPORTE"].map(bin_label_map)
    dfp["APOIO_MORADIA_LABEL"] = dfp["IN_APOIO_MORADIA"].map(bin_label_map)

    # garantir que apoios/financiamento sejam 0/1
    for col in [
        "IN_FINANCIAMENTO_ESTUDANTIL",
        "IN_APOIO_ALIMENTACAO",
        "IN_APOIO_TRANSPORTE",
        "IN_APOIO_MORADIA",
    ]:
        dfp[col] = pd.to_numeric(dfp[col], errors="coerce").fillna(0).astype(int)

    # remover linhas sem idade
    dfp = dfp[dfp["NU_IDADE"].notna()].copy()

    return dfp


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------


def run_eda(dfp: pd.DataFrame) -> None:
    """Gera gráficos básicos de EDA e salva na pasta de resultados."""

    # alvo
    plt.figure()
    sns.countplot(data=dfp, x="ALVO")
    plt.xticks([0, 1], ["Não concluiu", "Concluiu"])
    plt.title("Distribuição da variável alvo (conclusão)")
    savefig("eda_alvo_distribuicao.png")

    # idade
    plt.figure()
    sns.histplot(dfp["NU_IDADE"], bins=30, kde=True)
    plt.title("Distribuição das idades")
    plt.xlabel("Idade")
    savefig("eda_idade_histograma.png")

    plt.figure()
    sns.boxplot(x=dfp["NU_IDADE"])
    plt.title("Boxplot da idade")
    plt.xlabel("Idade")
    savefig("eda_idade_boxplot.png")

    def grafico_taxa(col: str, titulo: str, filename: str) -> None:
        taxa = dfp.groupby(col)["ALVO"].mean().reset_index()
        plt.figure()
        sns.barplot(data=taxa, x=col, y="ALVO")
        plt.title(titulo)
        plt.ylabel("Probabilidade de conclusão")
        plt.xticks(rotation=45)
        savefig(filename)

    # sexo
    plt.figure()
    sns.countplot(data=dfp, x="TP_SEXO")
    plt.title("Distribuição por sexo")
    savefig("eda_sexo_distribuicao.png")
    grafico_taxa("TP_SEXO", "Taxa de conclusão por sexo", "eda_sexo_taxa_conclusao.png")

    # raça/cor
    plt.figure()
    sns.countplot(data=dfp, x="TP_COR_RACA")
    plt.title("Distribuição por raça/cor")
    plt.xticks(rotation=45)
    savefig("eda_raca_distribuicao.png")
    grafico_taxa(
        "TP_COR_RACA",
        "Taxa de conclusão por raça/cor",
        "eda_raca_taxa_conclusao.png",
    )

    # turno
    plt.figure()
    sns.countplot(data=dfp, x="TP_TURNO")
    plt.title("Distribuição por turno")
    plt.xticks(rotation=45)
    savefig("eda_turno_distribuicao.png")
    grafico_taxa(
        "TP_TURNO",
        "Taxa de conclusão por turno",
        "eda_turno_taxa_conclusao.png",
    )

    # tipo de ingresso
    plt.figure()
    sns.countplot(data=dfp, x="TIPO_INGRESSO")
    plt.title("Distribuição por tipo de ingresso")
    savefig("eda_ingresso_distribuicao.png")
    grafico_taxa(
        "TIPO_INGRESSO",
        "Taxa de conclusão por tipo de ingresso",
        "eda_ingresso_taxa_conclusao.png",
    )

    # financiamentos e apoios
    apoios_label = [
        ("FINANCIAMENTO_LABEL", "Financiamento estudantil", "eda_financiamento"),
        ("APOIO_ALIM_LABEL", "Apoio alimentação", "eda_apoio_alimentacao"),
        ("APOIO_TRANS_LABEL", "Apoio transporte", "eda_apoio_transporte"),
        ("APOIO_MORADIA_LABEL", "Apoio moradia", "eda_apoio_moradia"),
    ]

    for col_label, nome, prefix in apoios_label:
        plt.figure()
        sns.countplot(data=dfp, x=col_label, order=["Não", "Sim"])
        plt.title(f"Distribuição: {nome}")
        plt.xlabel(nome)
        plt.ylabel("Quantidade de alunos")
        savefig(f"{prefix}_distribuicao.png")

        taxa = dfp.groupby(col_label)["ALVO"].mean().reset_index()
        plt.figure()
        sns.barplot(data=taxa, x=col_label, y="ALVO", order=["Não", "Sim"])
        plt.title(f"Taxa de conclusão × {nome}")
        plt.xlabel(f"{nome} (Não/Sim)")
        plt.ylabel("Probabilidade de conclusão")
        savefig(f"{prefix}_taxa_conclusao.png")


# ----------------------------------------------------------------------
# MODELAGEM
# ----------------------------------------------------------------------


def prepare_xy(dfp: pd.DataFrame):
    """Define X, y e o pré-processador de colunas numéricas/categóricas."""
    y = dfp["ALVO"].copy()

    X = dfp[
        [
            "NU_IDADE",
            "TP_SEXO",
            "TP_COR_RACA",
            "TP_TURNO",
            "TIPO_INGRESSO",
            "IN_FINANCIAMENTO_ESTUDANTIL",
            "IN_APOIO_ALIMENTACAO",
            "IN_APOIO_TRANSPORTE",
            "IN_APOIO_MORADIA",
        ]
    ].copy()

    cat_cols = ["TP_SEXO", "TP_COR_RACA", "TP_TURNO", "TIPO_INGRESSO"]
    num_cols = ["NU_IDADE"]

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="passthrough",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessador, cat_cols, num_cols


def train_and_evaluate(
    X_train, X_test, y_train, y_test, preprocessador
) -> pd.DataFrame:
    """Treina vários modelos de classificação e retorna tabela de métricas."""

    modelos = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "SVM Linear": LinearSVC(random_state=42),
        "Logística": LogisticRegression(max_iter=3000),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    resultados = []

    for nome, modelo in modelos.items():
        pipe = Pipeline([("prep", preprocessador), ("modelo", modelo)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        resultados.append(
            {
                "Modelo": nome,
                "Acurácia": accuracy_score(y_test, y_pred),
                "Precisão": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1": f1_score(y_test, y_pred, zero_division=0),
            }
        )

    resultados_df = pd.DataFrame(resultados).sort_values("F1", ascending=False)

    # salvar tabela
    path = os.path.join(RESULTS_DIR, "resultados_modelos.csv")
    resultados_df.to_csv(path, index=False)
    print(f"[CSV] Resultados dos modelos em {path}")

    # gráfico comparativo
    plt.figure(figsize=(10, 6))
    resultados_df.set_index("Modelo")[["Acurácia", "F1"]].plot(kind="bar", ax=plt.gca())
    plt.title("Comparação de modelos (Acurácia × F1)")
    plt.ylabel("Métrica")
    plt.xticks(rotation=45, ha="right")
    plt.legend(
        title="Métrica",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    savefig("modelos_comparacao_acuracia_f1.png")

    return resultados_df


def compute_feature_importance(
    X_train,
    y_train,
    preprocessador,
    cat_cols,
    num_cols,
) -> None:
    """Treina Random Forest e XGBoost e salva importância das variáveis."""

    # Random Forest
    pipe_rf = Pipeline(
        [
            ("prep", preprocessador),
            ("modelo", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    pipe_rf.fit(X_train, y_train)

    ohe = pipe_rf.named_steps["prep"].named_transformers_["cat"]
    cat_features = list(ohe.get_feature_names_out(cat_cols))

    all_features = cat_features + num_cols + [
        "IN_FINANCIAMENTO_ESTUDANTIL",
        "IN_APOIO_ALIMENTACAO",
        "IN_APOIO_TRANSPORTE",
        "IN_APOIO_MORADIA",
    ]

    import_rf = pipe_rf.named_steps["modelo"].feature_importances_

    df_imp_rf = (
        pd.DataFrame({"feature": all_features, "importance": import_rf})
        .sort_values("importance", ascending=False)
    )
    path_rf = os.path.join(RESULTS_DIR, "feature_importance_rf.csv")
    df_imp_rf.to_csv(path_rf, index=False)
    print(f"[CSV] Importância RF em {path_rf}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp_rf.head(15), x="importance", y="feature")
    plt.title("Importância das variáveis — Random Forest")
    plt.xlabel("importance")
    plt.ylabel("feature")
    savefig("feature_importance_random_forest.png")

    # XGBoost
    pipe_xgb = Pipeline(
        [
            ("prep", preprocessador),
            (
                "modelo",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe_xgb.fit(X_train, y_train)

    import_xgb = pipe_xgb.named_steps["modelo"].feature_importances_

    df_imp_xgb = (
        pd.DataFrame({"feature": all_features, "importance": import_xgb})
        .sort_values("importance", ascending=False)
    )
    path_xgb = os.path.join(RESULTS_DIR, "feature_importance_xgb.csv")
    df_imp_xgb.to_csv(path_xgb, index=False)
    print(f"[CSV] Importância XGBoost em {path_xgb}")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp_xgb.head(15), x="importance", y="feature")
    plt.title("Importância das variáveis — XGBoost")
    plt.xlabel("importance")
    plt.ylabel("feature")
    savefig("feature_importance_xgboost.png")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------


def main() -> None:
    ensure_results_dir()

    print("[INFO] Carregando dados...")
    df_pl = load_data(CSV_PATH)

    print("[INFO] Amostrando e convertendo para Pandas...")
    dfp = sample_to_pandas(df_pl)

    print("[INFO] Mapeando categorias...")
    dfp = map_categorical_columns(dfp)

    print("[INFO] Gerando EDA...")
    run_eda(dfp)

    print("[INFO] Preparando X e y...")
    X_train, X_test, y_train, y_test, preprocessador, cat_cols, num_cols = prepare_xy(
        dfp
    )

    print("[INFO] Treinando modelos...")
    resultados_df = train_and_evaluate(
        X_train, X_test, y_train, y_test, preprocessador
    )
    print(resultados_df)

    print("[INFO] Calculando importância das variáveis...")
    compute_feature_importance(
        X_train, y_train, preprocessador, cat_cols, num_cols
    )

    print("[OK] Pipeline concluído. Resultados em:", RESULTS_DIR)


if __name__ == "__main__":
    main()
