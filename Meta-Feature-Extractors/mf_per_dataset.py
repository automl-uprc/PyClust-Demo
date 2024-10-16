import pandas as pd
import numpy as np
from clustml import MFExtractor
import os

cvi = os.listdir(r"D:\GitHub_D\dataset-exhaustive-search\CVIs")
master = []
for dataset in cvi: #[1:]:
    print(dataset)
    dataset_path = os.path.join(r"D:\GitHub_D\dataset-exhaustive-search\datasets", dataset, "data.csv")
    df = pd.read_csv(dataset_path, header=None)
    mfe = MFExtractor(df=np.array(df))
    result = mfe.calculate_mf()
    for work in mfe.mf_papers:
        mf_names = [x for x in result if work in result[x]["included_in"]]
        mf_values = [result[x]["value"] for x in result if work in result[x]["included_in"]]
        if type(mf_values[0]) == dict:
            mf_names = list(mf_values[0].keys())
            mf_values = list(mf_values[0].values())
        master.append((dataset, work, mf_values))



for i in range(cosine_matrix.shape[0]):
    if np.count_nonzero(cosine_matrix[i]) == 1:
        print("opa")

cosine_matrix[34]


edge_list = extract_edges(similarity_matrix=cosine_matrix, dataset_name=dataset)
df.shape
cosine_matrix.shape