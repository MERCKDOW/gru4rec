import os
import shutil
import sys
import pandas as pd
import datetime
from google.cloud import bigquery
import subprocess
import glob
print(os.getcwd())

sys.path.insert(0,'./GRU4Rec_PyTorch_Official/')
#WEB_RECS_DERIVED.gru_training_data

project = "prod-analytics-recommend-c1jeg"
dataset = "RED_RECS"
table_prefix = "TRAINING_SET_TEST"
table_suffix = ''
#os.environ["GCLOUD_PROJECT"] = 'expanded-nebula-754'
os.environ["GCLOUD_PROJECT"] = 'prod-analytics-recommend-c1jeg'

import os.path
orig_cwd = os.getcwd()
import numpy as np
import json
import time
from collections import OrderedDict
import importlib
GRU4Rec = importlib.import_module('gru4rec_pytorch').GRU4Rec
import evaluation
import importlib.util
import joblib
import gc
os.chdir(orig_cwd)

#device = 'cuda:0'
sample_store_size= 10000000

import os
import shutil
import sys
from gru4rec_pytorch import SessionDataIterator
import torch
os.chdir(orig_cwd)
#os.environ["GCLOUD_PROJECT"] = 'expanded-nebula-754'
os.environ["GCLOUD_PROJECT"] = 'prod-analytics-recommend-c1jeg'

import gc
print(torch.cuda.is_available())


from google.cloud import aiplatform

# TODO: Replace with a GCS bucket you have write access to
# You can list buckets using: !gcloud storage buckets list --project prod-analytics-recommend-c1jeg
BUCKET_NAME = "cdow"
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipeline_root"

# Example of how to submit the job with the fix:
# job = aiplatform.PipelineJob(
#     display_name="gru4rec-pipeline",
#     template_path="gru4rec_pipeline.json", # Or your pipeline file
#     pipeline_root=PIPELINE_ROOT,           # <--- Add this argument
#     project="prod-analytics-recommend-c1jeg",
#     location="us-central1",                # Ensure location matches
# )

# job.submit()


def get_bq_training_table(table_suffix, gcs_location, data_models_path):
  query = """
          SELECT * FROM `prod-analytics-recommend-c1jeg.RED_RECS.GRU_TRAINING_DATA`
        """
#expanded-nebula-754.sandbox_crdow.GRU_Pytorch_PDP_1SELECT * FROM `prod-analytics-recommend-c1jeg.RED_RECS.GRU_TRAINING_DATA`
#query = """prod-analytics-recommend-c1jeg.RED_RECS.GRU_TRAINING_DATA
#SELECT * FROM `prod-analytics-recommend-c1jeg.RED_RECS.GRU_TRAINING_DATA` prod-analytics-recommend-c1jeg.RED_RECS.GRU_TRAINING_DATA
#prod-analytics-recommend-c1jeg.RED_RECS.GRU_Pytorch_PDP
#"""
  client = bigquery.Client()
  query_job = client.query(query)

# Convert the query result to a pandas DataFrame
  df = query_job.to_dataframe()
  df.to_csv(data_models_path + table_prefix + "_" + table_suffix + '.csv', index=False)


def load_data(fname, **args):

    with open(fname, 'rt', encoding="utf-8") as f:
        header = f.readline().strip().split('\t')

    if args["session_key"] not in header:
        print(args["session_key"])
        print('ERROR. The colmn specified for session IDs')
        sys.exit(1)
    if args["item_key"] not in header:
        print(args["item_key"])
        print('ERROR. The colmn specified for item IDs')
        sys.exit(1)
    if args["time_key"] not in header:
        print(args["time_key"])
        print('ERROR. The colmn specified for Time')
        sys.exit(1)
    print('Loading data from TAB separated file: {}'.format(fname))

    data = pd.read_csv(fname, sep='\t', usecols=[args["session_key"], args["item_key"],args["time_key"]], dtype={args["session_key"]:'int32', args["item_key"]:'str'})

    return data




def preprocess_training_data_tmstp(infile, outfile):
    data = pd.read_csv(infile)

    data.columns = ['TimeStr', 'SessionId', 'ItemId']


    print(data.head())
    print(data.dtypes)
    data['TimeStr'] = data['TimeStr'].astype(str)


    item_supports = data.groupby('ItemId').size()
    data = data[np.isin(data.ItemId, item_supports[item_supports>=4].index)]
    del item_supports
    gc.collect()

    data['Time'] = data['TimeStr'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d_%H_%M').timestamp())

    print("lambda applied")
    del(data['TimeStr'])

    session_lengths = data.groupby('SessionId').size()
    data = data[np.isin(data.SessionId, session_lengths[session_lengths>3].index)]
    del session_lengths
    gc.collect()


    #item_supports = data.groupby('ItemId').size()
    #data = data[np.in1d(data.ItemId, item_supports[item_supports>=2].index)]
    #session_lengths = data.groupby('SessionId').size()
    #data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-86400].index
    train = data[np.isin(data.SessionId, session_train)]

    del data
    gc.collect()

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(outfile, sep='\t', index=False)


def predict_gru(mtype, gru, original_train_data, recs_file_name, table_suffix,  device, top_n, data_models_path):
    data = pd.read_csv(original_train_data, sep='\t', usecols=[0,1,2], dtype={0:str, 1:str, 2:str})


    items = data[['ItemId']]
    del data
    gc.collect()
    grouped_items = items.groupby('ItemId')
    del items
    gc.collect()

    distinct_items = grouped_items.count()


    del grouped_items
    gc.collect()


    recs_file = open('/content/out_red_gru4rec.csv', "w")
    recs_file.close()
    recs_file = open('/content/out_red_gru4rec.csv', "a")


    id_map=gru.data_iterator.itemidmap
    id_map_swpapped = pd.Series(id_map.index.values, index=id_map).to_numpy()

    batch_size = 1
    H = []
    for i in range(len(gru.layers)):
        H.append(torch.zeros((batch_size, gru.layers[i]), requires_grad=False, device=gru.device, dtype=torch.float32))


    for row in distinct_items.itertuples():
        i = row.Index
        for h in H: h.detach_()
        in_idx = torch.from_numpy(np.array([id_map[row.Index]])).to(device)
        O = gru.model.forward(in_idx, H, None, training=False)

        oscores = O.T
        O_np = oscores.detach().cpu().numpy()
        O_np = O_np.reshape(-1)
        #top_n = 30
        ind = O_np.argsort()[-31:]
        ind = ind.reshape(-1)
        ind = ind[::-1]
        top_n = id_map_swpapped[ind]

        j = 0
        for t in top_n:
            if str(row.Index) != str(t):
                recs_file.write(str(row.Index) + "," + str(t) + "," + str(O_np[id_map[t]]) + "," + str(j+1) + "\n")
                j=j+1

    recs_file.close()



    recs_file_name = "/content/out_red_gru4rec.csv"
    print('--------made it!----------')
    project_id = 'prod-analytics-recommend-c1jeg'
    dataset_id = 'RED_RECS'
    table_id = 'GRU_Pytorch_PDP_temp'
    file_path = '/content/out_red_gru4rec.csv'

    #prod-analytics-recommend-c1jeg.RED_RECS.GRU_Pytorch_PDP
    schema = [
        bigquery.SchemaField("pkey", "INTEGER"),
        bigquery.SchemaField("recs", "INTEGER"),
        bigquery.SchemaField("score", "FLOAT"),
        bigquery.SchemaField("rank", "INTEGER"),
    ]

    #table_id = "prod-analytics-recommend-c1jeg.RED_RECS.gru4rec_ouput_"
    table_id = "prod-analytics-recommend-c1jeg.RED_RECS.output_red_1"

    client = bigquery.Client()

    job_config = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,  # Skip the header row in the CSV file
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    table_suffix = ''

    with open(file_path, "rb") as source_file:
        load_job = client.load_table_from_file(source_file, table_id, job_config=job_config)
        load_job.result()

        print(f"Loaded {load_job.output_rows} rows into {table_id}.")

    # add labels, create final tableGA_4_WEB_RECS_DERIVED.RED_LABELS   #GRU_RED_TRAINING_SET expanded-nebula-754.sandbox_crdow.GRU_RED_PRODUCT_LABELS
    table_suffix = ''
    
    final_query = "SELECT pkey, product_id AS recs, score, rank FROM (SELECT product_id AS pkey, recs, score, rank FROM `prod-analytics-recommend-c1jeg.RED_RECS.output_red_1`\
                    JOIN `prod-analytics-recommend-c1jeg.RED_RECS.GRU_PRODUCT_LABELS` \
                     ON pkey=pid) JOIN `prod-analytics-recommend-c1jeg.RED_RECS.GRU_PRODUCT_LABELS` ON recs=pid ORDER BY pkey, rank"


    print(final_query)
    client = bigquery.Client()
    query_config = bigquery.QueryJobConfig()
    query_config.destination = project_id + "." + dataset_id + "." + "GRU_Pytorch_PDP"
    query_config.write_disposition = 'WRITE_TRUNCATE'

    query_job = client.q

def __main__():
    
    recs_file = open('/content/out_red_gru4rec.csv', "w")
    recs_file.close()
    t11 = datetime.datetime.now()

    base_path = "/content/"

    f = open(base_path + 'gru4rec_config.json')
    data = json.load(f)
    args_model = data["model_args"]
    args_data = data["file_args"]
    f.close()

    #table_suffix = get_training_table_suffix()
    data_models_path = args_data["train_data_path"] + table_suffix + "/"

    if not os.path.exists(data_models_path):
        os.makedirs(data_models_path)

    #gcs_location = "gs://nkhan/gru4rec/" + table_suffix + "/"
    #gcs_files = gcs_location + table_prefix + "_" + table_suffix + "_*.csv"

    get_bq_training_table()#table_suffix, gcs_files, data_models_path)

    original_train_data = data_models_path + table_prefix + "_" + table_suffix + ".csv"
    processed_train_data = data_models_path + 'processed_' + table_prefix + "_" + table_suffix + ".csv"
    model_file_name = data_models_path + 'gru_' + args_data["mtype"] + '_model_' + table_suffix + ".pt"
    recs_file_name = data_models_path + 'gru_' + args_data["mtype"] + '_recs_' + table_suffix + ".csv"

    device = args_model["device"]
    d = preprocess_training_data_tmstp(original_train_data, processed_train_data)
    print("preprocessed")
    del d
    gc.collect()

    gru = GRU4Rec(device=device)
    gru.set_params(**args_model)
    data = load_data(processed_train_data,**args_data)
    del processed_train_data
    gc.collect()
    print("data loaded")

    data.to_csv(data_models_path + "data.csv", sep='\t', index=False)
    print("data to csv")



    print('-----train------')
    t0 = time.time()
    gru.fit(data, sample_cache_max_size=sample_store_size, item_key=args_data["item_key"], session_key=args_data["session_key"], time_key=args_data["time_key"])
    t1 = time.time()
    print('Total training time: {:.2f}s'.format(t1 - t0))


    # saving dataframes and non pyorch types
    # not handled by torch.save, but can be loaded back with joblib.load
    #joblib.dump(gru.data_iterator.itemidmap, data_models_path + 'itemidmap_' + table_suffix + '.pkl')
    #gru.savemodel(model_file_name)
    #print('-----saved------')

    device = args_model["device"]
    top_n = 31

    print('-----predict------')
    predict_gru(args_data["mtype"], gru, data_models_path + 'data.csv', recs_file_name, table_suffix, device, top_n, data_models_path)# gcs_location,
    print('-----predict------')

    #shutil.rmtree(data_models_path)
    #print("Removed " + data_models_path)

    t12 = datetime.datetime.now()
    print("Total Time = " + str(t12 - t11))

