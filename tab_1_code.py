import gradio.utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs, make_moons
import gradio as gr


def load_csv(file, csv_options):
    """

    Args:
        file (gradio.utils.NamedString):
        csv_options (list):

    Returns:

    """

    if 'Headers' in csv_options:
        headers = 0
    else:
        headers = None
    try:
        df_ = pd.read_csv(file.name, header=headers)
        if 'Scale Data' in csv_options:
            ms = MinMaxScaler()
            df_ = ms.fit_transform(df_)
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=True),
                f"<h1 style='text-align: center; color: blue;'> Data Loaded Successfully with shape: ({df_.shape[0]},"
                f"{df_.shape[1]}) !</h1>", df_)
    except Exception as e:
        return str(e), None


def generate_data(synthetic_method, no_instances, no_features):
    x = None
    if synthetic_method == 'Blobs':
        x, y = make_blobs(n_samples=int(no_instances), n_features=int(no_features))
    elif synthetic_method == 'Moons':
        x, y = make_moons(n_samples=no_instances)
    df_ = pd.DataFrame(x)

    return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(visible=False), gr.update(visible=True),
            f"<h1 style='text-align: center; color: blue;'> Data Generated Successfully with shape: "
            f"({x.shape[0]},{x.shape[1]}) !</h1>", df_)
