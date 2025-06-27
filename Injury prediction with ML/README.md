# Injury Prediction with Machine Learning

## Overview

This project implements a machine learning pipeline for predicting injury severity in traffic accidents. The system analyzes various factors such as victim demographics, safety equipment usage, vehicle characteristics, and accident conditions to predict whether a victim will be injured, not injured, or deceased.

## Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Exploratory Data Analysis**: Visualizations of accident patterns and injury distributions
- **Multiple ML Models**: Implementation of K-Nearest Neighbors, Gradient Boosting, and Random Forest classifiers
- **Model Evaluation**: Cross-validation, confusion matrices, and classification reports
- **Automated Pipeline**: End-to-end analysis with logging and result saving

## Project Structure

```
Injury prediction with ML/
├── Dataset.csv                          # Raw accident data (24MB)
├── injury_prediction_ml.py             # Main Python script
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── output/                             # Generated outputs (created automatically)
│   ├── accidents_by_year.png           # Accidents distribution visualization
│   ├── injury_distribution.png         # Injury severity visualization
│   ├── knn_confusion_matrix.png        # KNN model confusion matrix
│   ├── knn_classification_report.txt   # KNN model performance report
│   ├── knn_model.joblib               # Saved KNN model
│   ├── knn_preprocessor.joblib        # Saved KNN preprocessor
│   ├── gradient_boosting_confusion_matrix.png
│   ├── gradient_boosting_classification_report.txt
│   ├── gradient_boosting_model.joblib
│   ├── gradient_boosting_preprocessor.joblib
│   ├── random_forest_confusion_matrix.png
│   ├── random_forest_classification_report.txt
│   ├── random_forest_model.joblib
│   └── random_forest_preprocessor.joblib
└── injury_prediction.log               # Execution log file
```

## Dataset Description

The `Dataset.csv` file contains traffic accident data with the following features:

- **CASE_ID**: Unique case identifier
- **PARTY_NUMBER**: Party number in the accident
- **VICTIM_NUMBER**: Victim identifier
- **VICTIM_ROLE**: Role of victim (driver/passenger)
- **VICTIM_SEX**: Gender of victim
- **VICTIM_AGE**: Age of victim
- **VICTIM_DEGREE_OF_INJURY**: Injury severity (1-8 scale)
- **VICTIM_SEATING_POSITION**: Seating position in vehicle
- **VICTIM_SAFETY_EQUIP_1**: Primary safety equipment
- **VICTIM_SAFETY_EQUIP_2**: Secondary safety equipment
- **VICTIM_EJECTED**: Whether victim was ejected
- **COUNTY**: County where accident occurred
- **CITY**: City where accident occurred
- **ACCIDENT_YEAR**: Year of accident

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project**:
   ```bash
   # If using git
   git clone <repository-url>
   cd "Injury prediction with ML"
   
   # Or simply download and extract the files to your desired directory
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All dependencies installed successfully!')"
   ```

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
python injury_prediction_ml.py
```

This will:
1. Load and explore the dataset
2. Create visualizations
3. Clean and preprocess the data
4. Train three different ML models
5. Evaluate model performance
6. Save all results to the `output/` directory

### Expected Output

The script will display:
- Dataset information and sample data
- Model training progress
- Cross-validation results
- Final model performance summary
- Best performing model identification

### Output Files

After running the script, you'll find the following in the `output/` directory:

#### Visualizations
- `accidents_by_year.png`: Distribution of accidents by year
- `injury_distribution.png`: Distribution of injury severity levels

#### Model Results
For each model (KNN, Gradient Boosting, Random Forest):
- `{model}_confusion_matrix.png`: Confusion matrix visualization
- `{model}_classification_report.txt`: Detailed performance metrics
- `{model}_model.joblib`: Saved trained model
- `{model}_preprocessor.joblib`: Saved data preprocessor

#### Logs
- `injury_prediction.log`: Detailed execution log

## Model Details

### 1. K-Nearest Neighbors (KNN)
- **Purpose**: Baseline classification model
- **Optimization**: Cross-validation to find optimal k value (1-20)
- **Features**: Uses all preprocessed features

### 2. Gradient Boosting
- **Purpose**: Ensemble learning with boosting
- **Optimization**: Cross-validation to find optimal number of estimators (10-500)
- **Features**: Handles both numerical and categorical features

### 3. Random Forest
- **Purpose**: Ensemble learning with bagging
- **Optimization**: Cross-validation to find optimal number of estimators (50-400)
- **Features**: Robust to overfitting, handles feature interactions

## Data Preprocessing

The pipeline includes comprehensive data preprocessing:

1. **Data Cleaning**:
   - Remove redundant columns (CASE_ID, COUNTY, VICTIM_NUMBER, CITY)
   - Handle missing and invalid values
   - Standardize variable names

2. **Feature Engineering**:
   - Create target variable (STATE) from injury severity
   - Add derived features (VEHICLE_TYPE, POSITION)
   - Handle categorical variables

3. **Data Imputation**:
   - Numerical variables: Median imputation
   - Categorical variables: Most frequent value imputation

4. **Feature Scaling**:
   - Numerical features: StandardScaler
   - Categorical features: OneHotEncoder

## Performance Metrics

The models are evaluated using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Precision for each class
- **Recall**: Recall for each class
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## Customization

### Modifying Sample Size

To change the number of samples used for training:

```python
# In injury_prediction_ml.py, modify the prepare_data method call
X_train_processed, X_test_processed = self.prepare_data(sample_size=30000)  # Default: None (for the full dataset)
```

### Adding New Models

To add additional ML models, extend the `InjuryPredictionML` class:

```python
def train_new_model(self, X_train_processed, X_test_processed):
    """Train a new model."""
    # Your model implementation here
    pass
```

### Changing Data Path

To use a different dataset:

```python
# When creating the analyzer instance
analyzer = InjuryPredictionML(data_path="path/to/your/data.csv")
```

## Troubleshooting

### Common Issues

1. **Memory Error**: 
   - Reduce sample size in `prepare_data()` method
   - Ensure sufficient RAM (recommended: 8GB+)

2. **Missing Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Dataset Not Found**:
   - Ensure `Dataset.csv` is in the same directory as the script
   - Check file permissions

4. **Import Errors**:
   - Verify Python version (3.7+)
   - Reinstall dependencies: `pip install --force-reinstall -r requirements.txt`

### Performance Optimization

- **Faster Execution**: Reduce sample size or use fewer cross-validation folds
- **Better Accuracy**: Increase sample size or tune hyperparameters
- **Memory Usage**: Process data in chunks for very large datasets

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

For questions or support, please open an issue in the repository or contact the maintainers.

## Acknowledgments

- Dataset provided for educational purposes
- Built with scikit-learn, pandas, and other open-source libraries
- Inspired by traffic safety research and machine learning applications

---

**Note**: This project is for educational and research purposes. The models should not be used for real-world injury prediction without proper validation and medical oversight. 