import pandas as pd
from tqdm import tqdm

from src.utils.utils import get_logger
import logging

from pyspark.sql import SparkSession

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from pathlib import Path

from xgboost import XGBClassifier

RECURRENT_THRESHOLD = 4

MELT_VARS = ['key', 'ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf', 'ncm_le',
              'id_pais_origem_le', 'importador_municipio_le', 'urf_le', 'importador_uf_le', 'true_status',
              'predicted_status', 'prob_aumento', 'prob_manter', 'prob_queda']

logger = logging.getLogger(__name__)
logger = get_logger(logger=logger)

# Preparation constants
ESTADOS_BR = ["AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE",
              "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO", "DF"]
INDEX_COLUMNS = ["key","ncm", "id_pais_origem", "importador_municipio", "importador_uf", "urf"]
CATEGORICAL_COLUMNS = ['ncm', 'id_pais_origem', 'importador_municipio', 'urf', "importador_uf", "status_q1"]
COLUMNS_2_INTERPOLATE = ['201801', '201802', '201901', '201902', '202001', '202002', '202101', '202102', '202201',
                         '202202', '202301', '202302', '202401', '202402']
FEATURES_COLUMNS = ['202201', '202202', '202301', '202302', 'ncm_le', 'id_pais_origem_le', 'importador_municipio_le',
                    'urf_le', 'importador_uf_le']

# CONSTANTS_FOR_THE_MODEL
MODEL_OUTPUT_COLUMNS = ['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf',
                        "prob_aumento", "prob_manter", "prob_queda"]
FINAL_OUTPUT_COLUMNS = ['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf',
                        "prob_aumento", "prob_manter", "prob_queda", "ano_semestre", "avg_valor_unitario", "ano",
                        "semestre"]

SKIP_DATA_TREATMENT = True

def get_ups_n_downs(v_1, v_2):
    if v_1 < v_2:
        status = "aumento"
    elif v_1 > v_2:
        status = "queda"
    else:
        status = "manteve"
    return status


def correct_counties_n_states(df) -> pd.DataFrame:
    df.dropna(inplace=True)
    df = df[(df["importador_uf"] != "") & (df["importador_municipio"] != "")].copy()
    df["old_municipio"] = df["importador_municipio"]

    logger.info("\tajustando importador UF")
    df["importador_uf_new"] = df.apply(lambda x: x["importador_uf"] if x["importador_uf"] in ESTADOS_BR \
        else x["importador_municipio"], axis=1)

    logger.info("\tajustando importador municipio")
    df["importador_municipio_new"] = df.apply(lambda x: x["old_municipio"] if x["old_municipio"] not in ESTADOS_BR \
        else x["importador_uf"], axis=1)

    df.drop(columns=['importador_municipio', "importador_uf", "old_municipio"], inplace=True)
    df.rename(columns={"importador_municipio_new": "importador_municipio",
                       "importador_uf_new": "importador_uf"}, inplace=True)

    return df


def create_dates_template():
    years_df = pd.DataFrame.from_dict({"ano": [2018, 2019, 2020, 2021, 2022, 2023, 2024]})
    semesters_df = pd.DataFrame.from_dict({"semestre": [1, 2]})
    dates_template = years_df.join(semesters_df, how="cross")
    dates_template["ano_semestre"] = dates_template["ano"] * 100 + dates_template["semestre"]
    dates_template.drop(columns=["ano", "semestre"], inplace=True)

    return dates_template


def get_only_recurrent(df_recurrent):
    df_recurrent["key"] = df_recurrent["ncm"].astype(str) + '-' + df_recurrent["id_pais_origem"] + '-' + \
                          df_recurrent['importador_municipio'] + '-' + df_recurrent['urf']

    gp_data = df_recurrent.groupby(["key", "ano", "semestre"], as_index=False).mean("avg_valor_item")
    df_count = gp_data["key"].value_counts().reset_index()
    recurrent_keys = df_count[df_count["key"] >= RECURRENT_THRESHOLD]["index"]
    df_recurrent = df_recurrent[df_recurrent["key"].isin(recurrent_keys)].copy()

    return df_recurrent


def interpolate_values(data_in_path: str = '../data/interim/2_interpolate',
                       data_out_path: str = '../data/interim/interpolated_categorized/'):
    # ToDO: Automatically clean the output path
    # logger.info("Cleaning output path")

    data_dir = Path(data_in_path)
    files = [parquet_file for parquet_file in data_dir.glob('*.parquet')]
    with tqdm(total=len(files), desc="Interpolating and transforming values") as pbar:
        for file in files:
            df_aux = pd.read_parquet(file)
            df_aux.dropna(subset=['202302'], inplace=True)
            df_aux[COLUMNS_2_INTERPOLATE] = df_aux[COLUMNS_2_INTERPOLATE].interpolate(axis=1, method="linear")
            df_aux["status_q1"] = df_aux.apply(lambda x: get_ups_n_downs(x['202301'], x['202302']), axis=1)
            df_aux.to_parquet(f'{data_out_path}{file.name}')
            pbar.update(1)

    logger.info(f"Output path: {data_out_path}")
    return df_aux["status_q1"].value_counts() / df_aux.shape[0]


def encode_categorical(data_in_path: str = '../data/interim/interpolated_categorized/',
                       data_out_path: str = '../data/interim/ready_to_train/'):
    data_dir = Path(data_in_path)
    files = [parquet_file for parquet_file in data_dir.glob('*.parquet')]
    with tqdm(total=len(files), desc="Interpolating and transforming values") as pbar:
        for file in files:
            df_aux = pd.read_parquet(file)
            with tqdm(total=len(CATEGORICAL_COLUMNS), desc=f"\tfor file {file.name}") as pbar_s:
                for column in CATEGORICAL_COLUMNS:
                    le = LabelEncoder()
                    df_aux[column + "_le"] = le.fit_transform(df_aux[column])
                    pbar_s.update(1)
            df_aux.to_parquet(f'{data_out_path}{file.name}')
            pbar.update(1)


def melt_data(df_pivoted):

    df_pivoted = df_pivoted.reset_index()
    print(df_pivoted.columns)
    print(MELT_VARS)

    df_pivoted = df_pivoted.melt(id_vars=MELT_VARS)
    df_pivoted.rename(columns={"variable": "ano_semestre", "value": "avg_valor_unitario"}, inplace=True)
    df_pivoted["ano"] = df_pivoted["ano_semestre"].str[:4]
    df_pivoted["semestre"] = df_pivoted["ano_semestre"].str[-1]
    df_pivoted.dropna(subset="avg_valor_unitario", inplace=True)
    df_pivoted["avg_valor_unitario"] = df_pivoted["avg_valor_unitario"].astype(float)

    return df_pivoted


if __name__ == "__main__":
    logger.info("Inniting Spark Session")
    spark = SparkSession.builder.master("local[1]").appName("trend_proba").getOrCreate()

    to_interpolate_path = '../data/interim/2_interpolate'
    interpolated_path = '../data/interim/interpolated_categorized/'
    transformed_path = '../data/interim/ready_to_train/'
    train_data_path = '../data/interim/data_ready_to_train.parquet'

    # if SKIP_DATA_TREATMENT == False:
    # Leitura dos dados históricos
    logger.info("Lendo arquivo parquet")
    grouped_data = pd.read_parquet("../data/processed/average_unity_price_historic.parquet")

    logger.info("Filtrando e corrigindo dados")
    grouped_data = correct_counties_n_states(grouped_data)

    logger.info("\tcorrigindo NCM")
    grouped_data["ncm"] = grouped_data["ncm"].astype(float).astype(int).astype(str)

    logger.info("Criando dataframes de datas e chaves únicas")

    dates_template = create_dates_template()

    logger.info("\tcriação de chaves únicas")

    logger.info("\tseleção somente do que é recorrente")


    grouped_data = get_only_recurrent(grouped_data)
    grouped_data.to_parquet("../data/interim/grouped_data.parquet", index=False)



    unique_keys = grouped_data.drop_duplicates(subset="key")[INDEX_COLUMNS]
    cross_template = unique_keys.merge(dates_template, how="cross")

    cross_template.to_parquet("../data/interim/cross_template.parquet", index=False)

    # Passando operações para o Spark
    logger.info("Lendo arquivos com Spark")
    cross_template_sp = spark.read.parquet("../data/interim/cross_template.parquet")
    grouped_data = spark.read.parquet("../data/interim/grouped_data.parquet")

    logger.info("Cruzando template com valores históricos")
    grouped_data = grouped_data.groupBy(["key", "ano_semestre"]).avg("avg_valor_item")
    grouped_data = grouped_data.withColumnRenamed("avg(avg_valor_item)", "avg_valor_item")
    df_filled = cross_template_sp.join(grouped_data, on=["key", "ano_semestre"], how="left")

    logger.info("Transformando valores em categóricos")
    logger.info("\tpivotando valores")
    df_filled_pivot = df_filled.groupBy("key", "ncm", "id_pais_origem", "importador_municipio", "importador_uf",
                                        "urf").pivot("ano_semestre").avg("avg_valor_item")
    df_filled_pivot.write.parquet('../data/interim/2_interpolate', mode="overwrite")

    logger.info("\tinterpolando valores pivotados")
    to_interpolate_path = '../data/interim/2_interpolate'
    interpolated_path = '../data/interim/interpolated_categorized/'
    transformed_path = '../data/interim/ready_to_train/'
    train_data_path = '../data/interim/data_ready_to_train.parquet'

    categories_dist = interpolate_values(to_interpolate_path, interpolated_path)

    logger.debug("Distribuição dos status:")
    print(categories_dist)

    logger.info("Encodando dados categóricos")
    encode_categorical(interpolated_path, transformed_path)

    logger.info("Preparando dados para o treinamento do modelo")
        # df_inter = melt_data()


    df_inter = pd.read_parquet(transformed_path)

    logger.info("Separando treino e teste")
    X = df_inter.set_index(INDEX_COLUMNS)[FEATURES_COLUMNS]
    y = df_inter.set_index(INDEX_COLUMNS)["status_q1_le"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logger.info("Treinando modelo")
    xgbc = XGBClassifier(enable_categorical=True)
    xgbc.fit(X_train, y_train)

    logger.info("Realizando previsões")
    y_hat = xgbc.predict(X)
    y_hat_prob = xgbc.predict_proba(X)

    df_results = X.copy()
    df_results["true_status"] = y
    df_results["predicted_status"] = y_hat
    df_results["prob_aumento"] = y_hat_prob[:,0]
    df_results["prob_manter"] = y_hat_prob[:,1]
    df_results["prob_queda"] = y_hat_prob[:,2]

    df_results = melt_data(df_results)
    # df_inter["ano"] = df_inter["ano"].astype(int)
    # df_inter["semestre"] = df_inter["semestre"].astype(int)

    output_path = "../data/processed/trended_data_interpolated_treated.parquet"
    df_results = df_results[FINAL_OUTPUT_COLUMNS].copy()
    df_results.to_parquet(output_path)
    logger.info(f"Model results stored at {output_path}")

