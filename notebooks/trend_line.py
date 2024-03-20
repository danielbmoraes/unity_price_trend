
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from unity_price_trend.src.utils.utils import get_logger
import logging

logger = logging.getLogger(__name__)
logger = get_logger(logger=logger)

if __name__ == "__main__":
    # Reading Data
    grouped_data = pd.read_parquet("../../data/processed/average_unity_price_historic.parquet")

    logger.info("Filtrando e corrigindo dados")

    # Deleting all null values and empty spaces in important columns
    grouped_data.dropna(inplace=True)
    grouped_data = grouped_data[(grouped_data["importador_uf"] != "") \
                                & (grouped_data["importador_municipio"] != "")].copy()

    # Correction of states and counties
    estados_br = ["AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE",
                  "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO", "DF"]

    grouped_data["old_municipio"] = grouped_data["importador_municipio"]

    # Treating the data

    logger.info("\tajustando importador UF")
    grouped_data["importador_uf_new"] = grouped_data.apply(lambda x: x["importador_uf"] \
        if x["importador_uf"] in estados_br else x["importador_municipio"], axis=1)

    logger.info("\tajustando importador municipio")
    grouped_data["importador_municipio_new"] = grouped_data.apply(lambda x: x["old_municipio"] \
        if x["old_municipio"] not in estados_br else x["importador_uf"], axis=1)

    grouped_data = grouped_data.groupby(['ncm', 'importador_uf', 'importador_municipio', 'urf', 'id_pais_origem',
                                         'ano', 'semestre'], as_index=False).mean('avg_valor_item')

    logger.info("Criando dataframes de datas")
    # Creation of historic dates template
    years_df = pd.DataFrame.from_dict({"ano": [2018, 2019, 2020, 2021, 2022, 2023]})
    semesters_df = pd.DataFrame.from_dict({"semestre": [1, 2]})
    dates_template = years_df.join(semesters_df, how="cross")
    dates_template["ano_semestre"] = dates_template["ano"] * 100 + dates_template["semestre"]

    # Creation of dates template for the trend line
    new_data = {"ano": [2024, 2024],
                "semestre": [1, 2]}
    df_new_data = pd.DataFrame(new_data)
    df_new_data["ano_semestre"] = df_new_data["ano"] * 100 + df_new_data["semestre"]

    logger.info("Filtrando dataframe somente com chaves válidas não processadas")

    # Reading the data already processed (create a empty dataframe if theres no processed data)
    if os.path.isfile("../data/processed/trend_values/trend_lines.parquet"):
        df_total = pd.read_parquet("../data/processed/trend_values/trend_lines.parquet")

        # Creating the list of already processed and valid keys
        keys_processed = df_total[['id_pais_origem', 'ncm', 'importador_municipio', 'urf']].drop_duplicates()
        keys_processed["key"] = keys_processed["ncm"].astype(str) + '-' + keys_processed["id_pais_origem"] + '-' + \
                                keys_processed['importador_uf'] + '-' + keys_processed['importador_municipio'] + '-' + \
                                keys_processed['urf']

        already_processed = keys_processed["key"].to_list()

    else:
        df_total = pd.DataFrame()
        already_processed = []

    # Filtering only for the data in the last five years
    grouped_data = grouped_data[grouped_data["ano"] < 2024]
    grouped_data["key"] = grouped_data["ncm"].astype(str) + '-' + grouped_data["id_pais_origem"] + '-' + \
                          grouped_data['importador_uf'] + '-' + grouped_data['importador_municipio'] + '-' + \
                          grouped_data['urf']

    # Filtering the keys that is constantily repeated (so we could make a good trend line)
    count = pd.DataFrame(grouped_data["key"].value_counts())
    threshold_count = 4
    count = count[count["key"] >= threshold_count]
    keys_2_process = count.reset_index()["index"].to_list()

    # Filtering the dataset for keys not processed and recurrent keys
    grouped_data = grouped_data[~grouped_data["key"].isin(already_processed)]
    grouped_data = grouped_data[grouped_data["key"].isin(keys_2_process)]

    logger.info("Iniciando criacao da linha")
    # %%
    file_count = 0
    grouped = grouped_data.groupby(['id_pais_origem', 'ncm', 'importador_municipio', 'urf'])
    groups_qtd = grouped_data[['id_pais_origem', 'ncm', 'importador_municipio', 'urf']].drop_duplicates().shape[0]
    with tqdm(total=groups_qtd, desc="Criando linha de tendencia para preco unitario") as pbar:
        for key, df_group in grouped:

            df_aux_hist = grouped_data[
                (grouped_data['id_pais_origem'] == key[0]) &
                (grouped_data['ncm'] == key[1]) &
                # (grouped_data['importador_uf'] == key[2]) &
                (grouped_data['importador_municipio'] == key[2]) &
                (grouped_data['urf'] == key[3])
                ].groupby(["ano_semestre"], as_index=False).mean("avg_valor_item")

            if (df_aux_hist.shape[0] > 0) and (key[0] + '-' + key[1] not in already_processed):
                if len(df_aux_hist["ano_semestre"].unique()) < 5:
                    # Interpolate if it hasn't enough data to infer
                    gabarito_aux = dates_template.copy()
                    df_aux_hist = gabarito_aux.merge(df_aux_hist, on=['ano_semestre'], how='left')
                    df_aux_hist["avg_valor_item"] = df_aux_hist["avg_valor_item"].interpolate()
                    df_aux_hist.dropna(axis=0, inplace=True)

                df_aux_trend = df_new_data.copy()
                z = np.polyfit(df_aux_hist["ano_semestre"], df_aux_hist["avg_valor_item"], 1)
                p = np.poly1d(z)
                df_aux_trend["avg_valor_item"] = p(df_aux_trend["ano_semestre"])

                df_aux_trend['ncm'] = key[0]
                df_aux_trend['id_pais_origem'] = key[1]
                df_aux_trend['importador_municipio'] = key[2]
                df_aux_trend['urf'] = key[3]
                df_aux_trend["ano"] = df_aux_trend["ano"].astype(int)

                df_total = pd.concat([df_total, df_aux_trend])
                file_count += 1

                # For each 200 groups processed, it'll update the final dataframe
                if file_count % 200 == 0:
                    df_total.to_parquet(f"../data/processed/trend_values/trend_lines.parquet", index=False)

            pbar.update(1)

    # At the end, it should save at the end of execution
    df_total.to_parquet(f"../data/processed/trend_values/trend_lines.parquet", index=False)
