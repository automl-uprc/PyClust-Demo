import sys

import gradio
import numpy as np
import pandas as pd

sys.path.append(r"D:\GitHub_D\Meta-Feature-Extractors")
from clustml import MFExtractor

mfe = MFExtractor()
mf_categories = mfe.mf_categories
mf_papers = mfe.mf_papers
all_mf = [key for key in mfe.meta_features]


def update_search_type_dropdown(method):
    var = {"Category": mf_categories,
           "Paper": mf_papers,
           "Custom Selection": all_mf}
    return gradio.update(choices=var[method])


def calculate_mf(data, search_type, search_choices):
    search_choices = search_choices[0]
    name = None
    category = None
    included_in = None

    if search_type == "Category":
        category = search_choices
    elif search_type == "Paper":
        included_in = search_choices
    mfe_ = MFExtractor(data)
    mfe_.calculate_mf(name=name, category=category, included_in=included_in)
    return mfe_.search_mf(name=name, category=category, included_in=included_in, search_type="values")
