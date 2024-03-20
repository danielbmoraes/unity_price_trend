import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType

from unity_price_trend.src.utils.utils import get_logger

import logging

import pyarrow.parquet as pq

from tqdm import tqdm

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    logger = get_logger(logger=logger)

    logger.info("Inniting Spark Session")
    spark = SparkSession.builder.master("local[1]").appName("attributes_dict").getOrCreate()
    # logger.info("Spark Session info:")
    # ToDo: Try to print the Session infos

    logger.info("Reading historic data")
    historic_data = spark.read.parquet("../data/processed/average_unity_price_historic.parquet")

    logger.info("Creating historic template dates")
    template_data = [
        {"ano": 2018, "semestre": 1}, {"ano": 2018, "semestre": 2},
        {"ano": 2019, "semestre": 1}, {"ano": 2019, "semestre": 2},
        {"ano": 2020, "semestre": 1}, {"ano": 2020, "semestre": 2},
        {"ano": 2021, "semestre": 1}, {"ano": 2021, "semestre": 2},
        {"ano": 2022, "semestre": 1}, {"ano": 2022, "semestre": 2},
        {"ano": 2023, "semestre": 1}, {"ano": 2023, "semestre": 2},
        {"ano": 2024, "semestre": 1}, {"ano": 2024, "semestre": 2}]

    template_schema = StructType([StructField("ano", IntegerType()),
                                  StructField("semestre", IntegerType())])

    data_template = spark.createDataFrame(data=template_data, schema=template_schema)

    logger.info("Crossing possible combinations with the dates template")
    unique_combinations = historic_data.select(
        historic_data.ncm,
        historic_data.id_pais_origem,
        historic_data.importador_municipio,
        historic_data.importador_uf,
        historic_data.urf
    ).distinct()


    data_2b_filled = unique_combinations.crossJoin(data_template)
    # data_2b_filled = data_2b_filled.dropna()

    logger.info("\tDiscarding empty URFs, UFs and counties")
    data_2b_filled = data_2b_filled.filter((data_2b_filled.urf != "") &
                          (data_2b_filled.importador_uf != "") &
                          (data_2b_filled.importador_municipio != ""))
    #%%
    data_filled = data_2b_filled.join(historic_data, ["ncm", "id_pais_origem", "importador_municipio", "importador_uf", "urf", "ano", "semestre"], how='left')

    logger.info("Crossing the template with the trend line infos")
    tl_data = spark.read.parquet("../data/processed/trend_values/trend_lines.parquet")

    data_trended = data_filled.join(tl_data, ["ncm", "id_pais_origem"], how='left').select(data_filled.ncm,
                                                                                        data_filled.id_pais_origem,
                                                                                        data_filled.importador_municipio,
                                                                                        data_filled.importador_uf,
                                                                                        data_filled.urf,
                                                                                        data_filled.ano,
                                                                                        data_filled.semestre,
                                                                                        tl_data.avg_valor_item)

    logger.info("Saving the interim file")
    data_trended.write.parquet("../data/interim/trended_values/", mode="overwrite")
    spark.sparkContext.stop()

    logger.info("Starting interpolation")
    del data_trended, data_2b_filled, data_filled, data_template
    table = pq.read_table("../data/interim/trended_values/")
    df = table.to_pandas()

    file_count = 0
    inter_df = pd.DataFrame()
    grouped = df.groupby(['ncm', 'id_pais_origem', 'importador_municipio', 'importador_uf', 'urf','ano', 'semestre'])
    groups_qtd = df[['id_pais_origem', 'ncm', 'importador_municipio', 'importador_uf', 'urf']].drop_duplicates().shape[0]

    with tqdm(total=groups_qtd, desc="Interpolating missing values") as pbar:
        for key, df_group in grouped:
            df_group["avg_valor_item"] = df_group["avg_valor_item"].interpolate()
            inter_df = pd.concat([df_group, inter_df])

            file_count += 1

            # For each 200 groups processed, it'll update the final dataframe
            if file_count % 200 == 0:
                inter_df.to_parquet(f"../data/interim/interpolated_values.parquet", index=False)

            pbar.update(1)

    df = inter_df.copy()
    logger.info("Cleaning and preparing the dataset")
    logger.info("\tGrouping the data")
    df_grouped = df.groupby(['ncm', 'id_pais_origem', 'importador_municipio',
                             'importador_uf', 'urf','ano', 'semestre'], as_index=False).mean('avg_valor_item')

    df_grouped["old_municipio"] = df_grouped["importador_municipio"]

    br_states = ["AC", "AL", "AP", "AM", "BA", "CE", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE",
                  "PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO","DF"]

    logger.info("\tChecking for counties and states changes")
    logger.info("\t\tChecking state's names")
    df_grouped["importador_uf_new"] = df_grouped.apply(lambda x: x["importador_uf"] if x["importador_uf"] in br_states \
        else x["importador_municipio"], axis=1)

    logger.info("\t\tChecking countie's names")
    df_grouped["importador_municipio_new"] = df_grouped.apply(lambda x: x["old_municipio"] if x["old_municipio"] not in br_states \
        else x["importador_uf"],axis=1)

    df_grouped.drop(columns=['importador_municipio', "importador_uf", "old_municipio"], inplace=True)
    df_grouped.rename(columns={"importador_municipio_new": "importador_municipio",
                       "importador_uf_new": "importador_uf",
                        "id_pais_origem": "name_pt"}, inplace=True)

    df_grouped = df_grouped[df_grouped["avg_valor_item"] > 0]

    logger.info("Storing the data into the file '/processed/trend_line_and_historic_average_unity_price.parquet'")
    df_grouped.to_parquet("../data/processed/trend_line_and_historic_average_unity_price.parquet",index=False)
    logger.info("Preparation Done")