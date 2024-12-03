import json
import os

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pyclustkit.metalearning import  MFExtractor


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import LeaveOneOut

from collections import  Counter

import pickle

mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]



mfe.search_mf(category="descriptive", search_type="names")


def update_ml_options(selected_ml):
    """
    Visibility of meta-lerner classifier parameters
    Args:
        selected_ml (str):

    Returns:

    """
    if selected_ml == "KNN":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
    elif selected_ml == "DT":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def update_search_type_dropdown(method):
    var = {"Category": mf_categories,
           "Paper": mf_papers,
           "Custom Selection": all_mf}
    return gr.update(value="", choices=var[method])


def toggle_mf_selection_options(method):
    if method == "Custom Selection":
        return gr.update(visible=True), gr.update(visible=True)
    elif method == "All":
        return gr.update(visible=False), gr.update(visible=False)


def train_meta_learner(algorithm, mf, best_alg, *alg_options ):

    print(algorithm, mf , best_alg, alg_options)

    # Algorithm Config
    if algorithm == "KNN":
        alg = KNeighborsClassifier(n_neighbors=alg_options[0], metric=alg_options[1])

    elif algorithm == "DT":
        alg = DecisionTreeClassifier()

    # MF Config
    mfdf = pd.DataFrame()

    # Filter meta-features to include
    if mf == "Custom Selection" :
        mf_to_include = []

        if alg_options[3] == "Category" :
            for cat in alg_options[4]:
                mf_to_include += mfe.search_mf(category=cat, search_type="names")
        elif alg_options[3] == "Paper" :
            for paper in alg_options[4]:
                mf_to_include += mfe.search_mf(included_in=paper, search_type="names")

        for mf_ in os.listdir("repository/meta_features/synthetic_datasets"):
            with open(os.path.join(os.getcwd(), "repository/meta_features/synthetic_datasets", mf_), "r") as f:
                mf_dict = json.load(f)
                mf_dict = {k: v for k,v in mf_dict.items() if k in mf_to_include}

                mf_df = pd.json_normalize(mf_dict)
                mf_df['dataset'] = mf_.replace(".json", "").replace("_", "-")

                mfdf = pd.concat([mfdf,mf_df], ignore_index=True)

    mfdf = mfdf.reset_index(drop=True)



    # Get Labels
    with open("repository/best_alg_per_cvi/synthetic_datasets_best_alg.json", "r") as f:
        master_cvi_dict = json.load(f)
    best_alg_per_dataset = []
    if best_alg == "Most Popular Alg":
        for key in master_cvi_dict:
            count_cvi_pop = Counter(list(master_cvi_dict[key].values()))
            most_pop = count_cvi_pop.most_common(1)[0][0]
            best_alg_per_dataset.append(( key.replace(".json", "").replace("_", "-"), most_pop))

    elif best_alg == "Specific CVI":
        cvi_selected = alg_options[5]
        for key in master_cvi_dict:
            best_alg_per_dataset.append(( key.replace(".json", "").replace("_", "-"), master_cvi_dict[key][cvi_selected]))

    labels_df = pd.DataFrame(best_alg_per_dataset)
    labels_df.columns = ["dataset", "algorithm"]


    master_df = pd.merge(mfdf, labels_df, on="dataset", how="outer").reset_index(drop=True)
    master_df = master_df.fillna(0)
    print(mfdf.head(5))
    print(master_df[["dataset","algorithm"]])
    print(master_df.iloc[0])

    # Algorithm Training
    X = master_df.drop(columns=["dataset", "algorithm"])
    y = master_df["algorithm"]

    if master_df.shape[0] <= 40:
        loo = LeaveOneOut()
        y_true = []
        y_pred = []

        for train_index, test_index in loo.split(master_df):
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            alg.fit(x_train, y_train)

            y_true.append(y_test.iloc[0])  # Single test instance
            print(x_test)
            print(x_test.iloc[0])
            y_pred.append(alg.predict(x_test.iloc[0].values.reshape(1,-1))[0])

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
        disp.plot(cmap=plt.cm.Blues)
        plt.savefig("confusion_matrix_loocv.png")
        plt.close()

    gr.Info("Meta Learner trained successfully!")
    ml_meta_data  = {"no_datasets": master_df.shape[0], "evaluation": {"method": "Leave-One-Out", "accuracy": 10}, "meta-features":
        {"number": 10, "based_on": "category"}, "classes_that_appear_in_data": "" }

    return "confusion_matrix_loocv.png", ml_meta_data, alg

def save_meta_learner(model, model_name, model_metadata):
    """

    Args:
        model:
        model_name:

    Returns:

    """
    # ---(1)--- Check if model name is empty. If so, it's replaced with _ to avoid internal errors
    if model_name == "":
        model_name = "_"

    # ---(2)--- Check if any meta-learning model with the same name exists
    meta_learners_saved = os.listdir("repository/meta_learners/models")
    print(model_name + ".pkl" )
    print(meta_learners_saved)
    if model_name + ".pkl" in meta_learners_saved:
        print("pl")
        raise gr.Error("A meta-learner with the same ID is present in the repository.")

    # ---(3)--- Save the trained meta-learning model
    with open(f"repository/meta_learners/models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    gr.Info("Model saved Successfully!")

    # ---(4)--- Update meta-learner metadata in the repository
    print(model_metadata)
    with open(f"repository/meta_learners/meta-data.json", "r") as f:
        ml_metadata = json.load(f)

    ml_metadata.update({model_name:model_metadata})

    with open(f"repository/meta_learners/meta-data.json", "w") as f:
        json.dump(ml_metadata, f)

    gr.Info("Meta-Learner Metadata Repository Updated Successfully!")


# def refresh_meta_learner_repository(meta_learner_repo):
  #  meta_learners =