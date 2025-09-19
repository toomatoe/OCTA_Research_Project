# Machine Learning Model Comparison and Data Visualization

This repository contains Python scripts for machine learning model comparison and data visualization.

## Project Structure

```
├── ml_model_comparison.py    # Compare multiple ML models on a dataset
├── data_visualization.py     # Visualize model performance data
├── data/                     # Data directory (CSV files go here)
│   ├── README.md            # Instructions for data files
│   ├── INL.csv              # Dataset for ML model comparison (not included)
│   └── resultsTable.csv     # Model results data (not included)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Scripts Description

### ml_model_comparison.py
This script compares multiple machine learning models on a dataset:
- **Models tested**: Logistic Regression, SVM, KNN, Random Forest, Decision Tree, Naive Bayes, Gradient Boosting
- **Features**: 5-fold cross-validation, confusion matrix analysis, accuracy scoring
- **Output**: Validation and test accuracy for each model, best performing model

### data_visualization.py
This script creates visualizations of model performance data:
- **Features**: Bar plots of model accuracy by type
- **Libraries**: Uses matplotlib and seaborn for plotting
- **Output**: Visual comparison of average validation accuracy by model type

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - On Windows:
     ```cmd
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

Before running the scripts, you need to provide your own data files:

1. Place `INL.csv` in the `data/` directory for `ml_model_comparison.py`
2. Place `resultsTable.csv` in the `data/` directory for `data_visualization.py`

See `data/README.md` for more details about the expected data format.

## Usage

### Running the ML Model Comparison
```bash
python ml_model_comparison.py
```

This will:
- Load the dataset from `data/INL.csv`
- Train and evaluate 7 different ML models
- Display cross-validation scores and test accuracies
- Identify the best performing model

### Running the Data Visualization
```bash
python data_visualization.py
```

This will:
- Load results data from `data/resultsTable.csv`
- Create a bar plot showing average validation accuracy by model type
- Display the visualization

## Dependencies

- pandas (≥1.3.0) - Data manipulation and analysis
- scikit-learn (≥1.0.0) - Machine learning library
- matplotlib (≥3.5.0) - Plotting library
- seaborn (≥0.11.0) - Statistical visualization
- numpy (≥1.21.0) - Numerical computing

## Notes

- The virtual environment (`.venv/`) is excluded from version control
- Data files are not included in the repository for privacy/size reasons
- Make sure your data files match the expected format before running the scripts
- The scripts assume the target variable is in column '0' for the ML comparison

## Contributing

Feel free to fork this repository and submit pull requests for any improvements.

## License

[Add your license information here]