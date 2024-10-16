import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def get_hyperparameters(model_choice):
    if model_choice == "Logistic Regression":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif model_choice == "Decision Tree":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def train_and_visualize_model(file, target_column, model_choice, C, max_iter, max_depth, min_samples_split):
    # Read the uploaded CSV file
    df = pd.read_csv(file.name)

    # Split the dataset
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select and train the model
    if model_choice == "Logistic Regression":
        model = LogisticRegression(C=C, max_iter=max_iter)
    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    else:
        return "Invalid model choice", None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    cm_plot = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    cm_fig = cm_plot.get_figure()
    cm_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    cm_fig.savefig(cm_file.name)
    plt.close(cm_fig)

    return f"Model: {model_choice}\n\nAccuracy: {accuracy:.4f}\n\nClassification Report:\n{report}", cm_file.name


# Create a Gradio interface
def build_interface():
    with gr.Blocks() as interface:
        with gr.Column():
            file_input = gr.File(label="Upload CSV")
            target_column = gr.Textbox(label="Target Column")
            model_choice = gr.Radio(choices=["Logistic Regression", "Decision Tree"], label="Choose Model")

            # Hyperparameters inputs
            C = gr.Slider(0.01, 10.0, 1.0, label="C", visible=False)
            max_iter = gr.Slider(100, 1000, 200, label="Max Iterations", visible=False)
            max_depth = gr.Slider(1, 20, 5, label="Max Depth", visible=False)
            min_samples_split = gr.Slider(2, 20, 2, label="Min Samples Split", visible=False)

            model_choice.change(get_hyperparameters, inputs=model_choice,
                                outputs=[C, max_iter, max_depth, min_samples_split])

            model_results = gr.Textbox(label="Model Results")
            cm_output = gr.Image(label="Confusion Matrix")

            train_button = gr.Button("Train Model")
            train_button.click(train_and_visualize_model,
                               inputs=[file_input, target_column, model_choice, C, max_iter, max_depth,
                                       min_samples_split], outputs=[model_results, cm_output])

    return interface


# Launch the Gradio interface
if __name__ == "__main__":
    interface = build_interface()
    interface.launch()
