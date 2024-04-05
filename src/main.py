
import pandas as pd
from tqdm import tqdm

from src.utils.utils import get_logger
import logging

import os, shutil

from pyspark.sql import SparkSession

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from pathlib import Path

from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
logger = get_logger(logger=logger)


RECURRENT_THRESHOLD= 15

MELT_VARS = ['key', 'ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf', 'ncm_le',
              'id_pais_origem_le', 'importador_municipio_le', 'urf_le', 'importador_uf_le', 'true_status',
              'predicted_status', 'prob_aumento', 'prob_manter', 'prob_queda']


# Preparation constants
ESTADOS_BR = ["AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE",
              "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO", "DF"]
INDEX_COLUMNS = ["key","ncm", "id_pais_origem", "importador_municipio", "importador_uf", "urf"]
CATEGORICAL_COLUMNS = ['ncm', 'id_pais_origem', 'importador_municipio', 'urf', "importador_uf", "status_q1"]
COLUMNS_2_INTERPOLATE = ['201801', '201802', '201803', '201804',
                         '201901', '201902', '201903', '201904',
                         '202001', '202002', '202003', '202004',
                         '202101', '202102', '202103', '202104',
                         '202201', '202202', '202203', '202204',
                         '202301', '202302', '202303', '202304',
                         '202401']
FEATURES_COLUMNS = ['202201', '202202', '202301', '202302', 'ncm_le', 'id_pais_origem_le', 'importador_municipio_le',
                    'urf_le', 'importador_uf_le']

# CONSTANTS_FOR_THE_MODEL
MODEL_OUTPUT_COLUMNS = ['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf',
                        "prob_aumento", "prob_manter", "prob_queda"]
SKIP_DATA_TREATMENT = True

def clean_folder(folder_path: str) -> None:
    """
    The clean_folder function takes a folder path as an argument and deletes all files in that folder.
    It is used to clean the output directory before running the script.

    :param folder_path: str: Specify the folder to be cleaned
    :return: Nothing
    """
    logger.info(f"Cleaning folder '{folder_path}'")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error('Failed to delete %s. Reason: %s' % (file_path, e))

def get_ups_n_downs(previous_value, next_value):
    """
    The get_ups_n_downs function takes two values and returns a string indicating whether the first value is greater
    than, less than or equal to the next.

    :param previous_value: Is the price value for the previous quarter to be compared
    :param next_value: Is the price value for the next quarter to be compared
    :return: A string with the category of the variation
    """
    if previous_value < next_value:
        status = "aumento"
    elif previous_value > next_value:
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
    semesters_df = pd.DataFrame.from_dict({"semestre": [1, 2, 3, 4]})
    dates_template = years_df.join(semesters_df, how="cross")
    dates_template["ano_trimestre"] = dates_template["ano"] * 100 + dates_template["semestre"]
    dates_template.drop(columns=["ano", "semestre"], inplace=True)

    return dates_template


def get_only_recurrent(df_recurrent):
    df_recurrent["key"] = df_recurrent["ncm"].astype(str) + '-' + df_recurrent["id_pais_origem"] + '-' + \
                          df_recurrent['importador_municipio'] + '-' + df_recurrent['urf']

    gp_data = df_recurrent.groupby(["key", "ano", "trimestre"], as_index=False).mean("avg_valor_item")
    df_count = gp_data["key"].value_counts().reset_index()
    recurrent_keys = df_count[df_count["key"] >= RECURRENT_THRESHOLD]["index"]
    df_recurrent = df_recurrent[df_recurrent["key"].isin(recurrent_keys)].copy()

    return df_count, df_recurrent


def interpolate_values(data_in_path: str = '../data/interim/2_interpolate',
                       data_out_path: str = '../data/interim/interpolated_categorized/'):
    # ToDO: Automatically clean the output path
    # logger.info("Cleaning output path")
    clean_folder(data_out_path)
    data_dir = Path(data_in_path)
    files = [parquet_file for parquet_file in data_dir.glob('*.parquet')]
    with tqdm(total=len(files), desc="Interpolating and transforming values") as pbar:
        for file in files:
            df_aux = pd.read_parquet(file)
            df_aux.dropna(subset=['202302'], inplace=True)
            df_aux[COLUMNS_2_INTERPOLATE] = df_aux[COLUMNS_2_INTERPOLATE].interpolate(axis=1, method="linear")
            df_aux["status_q1"] = df_aux.apply(lambda x: get_ups_n_downs(x['202303'], x['202304']), axis=1)
            df_aux.to_parquet(f'{data_out_path}{file.name}')
            pbar.update(1)

    logger.info(f"Output path: {data_out_path}")
    return df_aux["status_q1"].value_counts() / df_aux.shape[0]


def encode_categorical(data_in_path: str = '../data/interim/interpolated_categorized/',
                       data_out_path: str = '../data/interim/ready_to_train/'):
    clean_folder(data_out_path)
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
    df_pivoted = df_pivoted.melt(id_vars=MELT_VARS)
    df_pivoted.rename(columns={"variable": "ano_trimestre", "value": "avg_valor_unitario"}, inplace=True)
    df_pivoted["ano"] = df_pivoted["ano_trimestre"].str[:4]
    df_pivoted["semestre"] = df_pivoted["ano_trimestre"].str[-1]
    df_pivoted.dropna(subset="avg_valor_unitario", inplace=True)
    df_pivoted["avg_valor_unitario"] = df_pivoted["avg_valor_unitario"].astype(float)

    return df_pivoted

if __name__ == "__main__":
    logger.info("Inniting Spark Session")
    spark = SparkSession.builder.master("local[1]").appName("trend_proba").getOrCreate()

    logger.info("Lendo arquivo parquet")
    grouped_data = pd.read_parquet("../data/processed/average_unity_price_historic.parquet")

    logger.info("Filtrando e corrigindo dados")
    grouped_data = correct_counties_n_states(grouped_data)

    logger.info("\tcorrigindo NCM")
    grouped_data["ncm"] = grouped_data["ncm"].astype(float).astype(int).astype(str)
    #%%
    logger.info("Criando dataframes de datas e chaves únicas")

    dates_template = create_dates_template()

    logger.info("\tcriação de chaves únicas")

    logger.info("\tseleção somente do que é recorrente")


    count, grouped_data = get_only_recurrent(grouped_data)
    grouped_data.to_parquet("../data/interim/grouped_data.parquet", index=False)

    unique_keys = grouped_data.drop_duplicates(subset="key")[INDEX_COLUMNS]
    cross_template = unique_keys.merge(dates_template, how="cross")

    cross_template.to_parquet("../data/interim/cross_template.parquet", index=False)

    # Passando operações para o Spark
    logger.info("Lendo arquivos com Spark")
    cross_template_sp = spark.read.parquet("../data/interim/cross_template.parquet")
    grouped_data = spark.read.parquet("../data/interim/grouped_data.parquet")

    logger.info("Cruzando template com valores históricos")
    grouped_data = grouped_data.groupBy(["key", "ano_trimestre"]).avg("avg_valor_item")
    grouped_data = grouped_data.withColumnRenamed("avg(avg_valor_item)", "avg_valor_item")
    df_filled = cross_template_sp.join(grouped_data, on=["key", "ano_trimestre"], how="left")

    logger.info("Transformando valores em categóricos")
    logger.info("\tpivotando valores")
    df_filled_pivot = df_filled.groupBy("key", "ncm", "id_pais_origem", "importador_municipio", "importador_uf",
                                        "urf").pivot("ano_trimestre").avg("avg_valor_item")
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


    output_path = "../data/processed/trended_data_interpolated_treated.parquet"
    df_results = df_results[MODEL_OUTPUT_COLUMNS].copy()
    df_results.to_parquet(output_path, index=False)
    logger.info(f"Model results stored at {output_path}")

    logger.info("Formatting Output")
    df_results = spark.read.parquet("../data/processed/trended_data_interpolated_treated.parquet")
    df_historic = spark.read.parquet("../data/processed/average_unity_price_historic.parquet")
    # df_historic.groupBy(df_historic.anomes).count().show()
    df_output = df_historic.join(df_results, on=['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf'], how="left").distinct()

    df_output = df_output.dropna(subset=['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf'])
    output_path = "../data/processed/output_formated_data/"
    df_output.write.parquet(output_path, mode="overwrite")
    pd.read_parquet(output_path).to_parquet("../data/processed/output_formated_data.parquet")