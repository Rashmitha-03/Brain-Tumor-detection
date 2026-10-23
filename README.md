# Brain Tumor Detection

A collection of notebooks, datasets, and a Streamlit application for experimenting with deep-learning-based brain MRI classification.

> **Medical disclaimer:** This project is for learning and research purposes only. Its predictions are not a diagnosis and must not be used to make medical decisions. Consult a qualified healthcare professional about medical imaging results.

## Project contents

| File | Description |
| --- | --- |
| `app.py` | Streamlit interface for uploading an MRI image and requesting a prediction from a selected model. |
| `img_classification.py` | Image resizing, model loading, preprocessing, and prediction helpers used by the app. |
| `Brain_Tumor_Classification_Using_Deep_Learning_Algorithms.ipynb` | Example VGG16 training notebook. Its introductory notes state that the model and reported scores are illustrative. |
| `TL_InceptionResNetV2.ipynb` | Notebook for training and evaluating a four-class InceptionResNetV2 transfer-learning model. |
| `brain_tumor_dataset.csv` | Tabular image-related data and MGMT labels. |
| `rsna_dataset.csv` | Tabular image-related data for RSNA patient IDs. |
| `train_labels.csv` | Patient IDs and MGMT-value labels. |
| `stacked_data1.npz` | NumPy archive containing prepared data. |
| `resnet.h5` | Model artifact saved by the InceptionResNetV2 notebook. |
| `Comparision-brain tumor.xlsx` | Spreadsheet comparing brain-tumor models or results. |

The notebooks and CSV/archive files cover different modeling workflows. The Streamlit app is a separate binary tumor/non-tumor workflow; the included `resnet.h5` is a four-class model and is **not a drop-in model for the app**.

## Requirements

- Python 3.9 or later, using a Python version supported by the installed TensorFlow release
- TensorFlow
- Streamlit
- Pillow
- NumPy

Install the packages for the app in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install streamlit tensorflow pillow numpy
```

The notebooks may require additional packages, including OpenCV, pandas, scikit-learn, Matplotlib, Seaborn, and tqdm. Install those in the notebook environment as needed.

## Running the app

From the project folder:

```bash
streamlit run app.py
```

The app expects compatible **binary** Keras model files named `model_vgg.h5` and `model_resnet.h5` in the project folder. Those files are not included here. Train or obtain models compatible with the preprocessing and output expected by `img_classification.py` before selecting either model. In particular, the included `resnet.h5` is a four-class InceptionResNetV2 artifact and should not be renamed and used as the app's ResNet model.

The app also expects `results_vgg.jpg` and `results_resnet.jpg` if the “Results from trained model” sidebar option is selected. Those image files are not included.

## Notebooks and data

Open either `.ipynb` file in Jupyter or Google Colab and follow its cells. The VGG16 example notebook describes its results as illustrative. `TL_InceptionResNetV2.ipynb` uses local, machine-specific training and testing paths; update those paths to point to your dataset before running it. The CSV files and NumPy archive in this folder may not replace the original image-folder datasets expected by the notebooks.

## Uploading to GitHub

Check file sizes before pushing. GitHub blocks regular Git files larger than 100 MB; the included `resnet.h5` is substantially larger than that limit, and `stacked_data1.npz` is also around the limit. Store large artifacts with [Git LFS](https://git-lfs.com/) or publish them as downloadable releases/external assets, and document where to obtain them. Do not commit the macOS `._*` metadata files.

If using Git LFS, install it and track large model/data artifacts before adding them:

```bash
git lfs install
git lfs track "*.h5" "*.npz"
git add .gitattributes
```

Then add and push the files you intend to publish. Review the dataset and model licenses and usage terms before redistributing them.
