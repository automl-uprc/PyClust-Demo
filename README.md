![PyClustUI](artifacts/pyclustui_logo.jpg)

# PyClustUI


## ✏️ Overview
PyClustUI is a graphical tool for meta-learning repository creation for clustering harnessing the capabilities of the python library **pyclustkit**. Additionally it encompasses various utilities for not only populating the meta-learning repository with a few clicks but also methods to train meta-learners on top of it capable of algorithm selection, one of the most popular meta-learning tasks. 

- **Meta Learning** : Use one of the pretrained meta-learners for algorithm selection or train your own based on a rich 
set of state-of-the art meta-features.

- **Model Search**: Out of the box grid search implementation with easy to define parameter space in the graphical 
interface 45 internal Cluster Validity Indices (CVIs) as developed in the pyclustkit library. 

PyClust is built as a Python library that can be installed through PyPI, but can also be used through a Gradio-based user interface and Docker image.

## 📓 Requirements

The demo is based upon certain python libraries which are listed in the requirements.txt file. 
It is basically built on

- Gradio 
- pyclustkit

The main software needed are:

- Docker
- Python>=3.12


## 🔁 Installation

There are two ways that you can use PyClustUI:


- ✅ By cloning the GitHub repository:

You can clone this repository with the following commands
```commandline
git clone https://github.com/your-username/PyClustUI.git  
cd PyClust-Demo   
```

and run main.py to use the Gradio-based user interface
```commandline
python main.py
```

- 🐳 Through Docker:

You can build and run the image with the following in CLI, assuming Docker is installed and running.
```commandline 
docker run -p 7861:7861 giannispoy/pyclust
```
After the successful run of the image, open a browser and go to
```commandline
localhost:7861
```
and the user interface will be accessible
![App Screenshot](artifacts/pyclustui_logo.jpg)


## 🤝 Contributions/Contact
Contributions are welcome! Please open an issue or submit a pull request if you'd like to improve the project.

Mail us  automl.uprc@gmail.com

