# Power System Load Type Prediction

## Overview
This project uses machine learning to predict the load type of a power system based on historical energy consumption and electrical measurements.

The model classifies power consumption into three categories:
- Light_Load
- Medium_Load
- Maximum_Load

## Objectives
- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Train and compare classification algorithms.
- Evaluate models using a chronological test set.
- Identify the best-performing model using weighted F1-score.

## Dataset
The dataset contains 35,041 rows and 9 columns.

Features include:
- Date_Time
- Usage_kWh
- Lagging_Current_Reactive.Power_kVarh
- Leading_Current_Reactive_Power_kVarh
- CO2(tCO2)
- Lagging_Current_Power_Factor
- Leading_Current_Power_Factor
- NSM

Target: `Load_Type`

**Dataset setup:** Place your CSV file in the `data/` directory as `load_data.csv`. The dataset is not included in this repository unless explicitly provided.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Jupyter Notebook

## Machine Learning Models
The project compares:
1. Logistic Regression
2. Random Forest Classifier
3. Extra Trees Classifier

The best model is selected using weighted F1-score on the test set.

## Validation Strategy
The data is sorted chronologically. All records from the latest calendar month are reserved for testing, while earlier records are used for training.

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Results
Latest recorded evaluation:
- Accuracy: 89%
- Weighted Precision: 90%
- Weighted Recall: 89%
- Weighted F1-score: 89%
- Test samples: 2,977

Results may vary if the dataset or training configuration changes.

## Project Structure
```text
load-type-prediction/
├── data/
├── src/
│   ├── train.py
│   └── predict.py
├── outputs/
├── load_type_eda.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Usage

### 1. Clone the repository
```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_NAME
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate it on Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Place `load_data.csv` in the `data/` directory.

### 5. Train and evaluate
```bash
python src/train.py
```

The evaluation reports, confusion matrix, model comparison and trained model are saved in the `outputs/` directory.

## License
Add a license if you intend to distribute this project for reuse.
