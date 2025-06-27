"""
Injury Prediction with Machine Learning
======================================

This script performs injury prediction analysis on accident data using various
machine learning algorithms including K-Nearest Neighbors, Gradient Boosting,
and Random Forest Classifiers.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Setup matplotlib
plt.ioff()
plt.rcParams["figure.figsize"] = (8, 6)
sns.set_style("whitegrid")

class InjuryPredictionML:
    def __init__(self, data_path="Dataset.csv", output_dir="output"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        
        logger.info(f"Initialized with data path: {data_path}")
    
    def load_data(self):
        """Load the dataset."""
        try:
            logger.info("Loading dataset...")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded. Shape: {self.df.shape}")
            return True
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return False
    
    def explore_data(self):
        """Perform initial data exploration."""
        logger.info("Starting data exploration...")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            print("Dataframe is not loaded.")
            return
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data types:\n{self.df.dtypes}")
        print(f"Missing values:\n{self.df.isnull().sum()}")
        print("\nSample data:")
        print(self.df.sample(n=5))
    
    def visualize_accidents_by_year(self):
        """Visualize accidents distribution by year."""
        logger.info("Creating accidents by year visualization...")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            return
        accident_year = self.df['ACCIDENT_YEAR']
        year_count = accident_year.value_counts()
        plt.figure(figsize=(10, 6))
        plt.bar(list(year_count.index), list(year_count.values), color='skyblue', edgecolor='black')
        plt.xticks(list(year_count.index), rotation=45)
        plt.ylabel("Number of Accidents")
        plt.xlabel("Year")
        plt.title("Distribution of Accidents by Year")
        plt.tight_layout()
        plt.savefig(self.output_dir / "accidents_by_year.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Accidents by year plot saved")
    
    def visualize_injury_distribution(self):
        """Visualize injury severity distribution."""
        logger.info("Creating injury distribution visualization...")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            return
        degree_of_injury = self.df['VICTIM_DEGREE_OF_INJURY']
        injury_count = degree_of_injury.value_counts()
        plt.figure(figsize=(10, 6))
        plt.bar(list(injury_count.index), list(injury_count.values), color='lightcoral', edgecolor='black')
        plt.xticks(list(injury_count.index))
        plt.xlabel("Injury Severity Degree")
        plt.ylabel("Number of Accidents")
        plt.title("Distribution of Injury Severity")
        plt.tight_layout()
        plt.savefig(self.output_dir / "injury_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Injury distribution plot saved")
    
    def clean_data(self):
        """Clean and preprocess the data."""
        logger.info("Starting data cleaning process...")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            return
        # Drop redundant variables
        variables_to_drop = ['CASE_ID', 'COUNTY', 'VICTIM_NUMBER', 'CITY']
        self.df = self.df.drop(columns=variables_to_drop, axis=1)
        
        # Rename variables
        new_variables_names = {
            'PARTY_NUMBER': 'PARTY_NUM',
            'VICTIM_ROLE': 'ROLE',
            'VICTIM_SEX': 'SEX',
            'VICTIM_AGE': 'AGE',
            'VICTIM_DEGREE_OF_INJURY': 'INJ_SEV',
            'VICTIM_SEATING_POSITION': 'SEAT_POS',
            'VICTIM_SAFETY_EQUIP_1': 'EQUIP_1',
            'VICTIM_SAFETY_EQUIP_2': 'EQUIP_2',
            'VICTIM_EJECTED': 'EJECTED',
            'ACCIDENT_YEAR': 'YEAR'
        }
        self.df = self.df.rename(columns=new_variables_names)
        
        # Replace missing/invalid values with NaN
        values_to_replace = {
            'SEAT_POS': ['-', '9'],
            'SEX': '-',
            'AGE': 998,
            'EJECTED': ['-', ' ', '3', '4'],
            'EQUIP_1': ['-', ' ', 'B'],
            'EQUIP_2': ['-', ' ', 'B']
        }
        self.df.replace(values_to_replace, np.nan, inplace=True)
        
        # Clean specific variables
        self.df.loc[~self.df['SEX'].isin(['M', 'F', np.nan]), 'SEX'] = np.nan
        self.df.loc[self.df['AGE'] > 80, 'AGE'] = 80
        
        # Clean EQUIP_1 and EQUIP_2
        valid_values = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                       'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                       'U', 'V', 'W', 'X', 'Y', pd.NA]
        self.df.loc[~self.df['EQUIP_1'].isin(valid_values), 'EQUIP_1'] = np.nan
        self.df.loc[~self.df['EQUIP_2'].isin(valid_values), 'EQUIP_2'] = np.nan
        
        # Clean EJECTED
        valid_values_ejected = ['0', '1', '2', '3', pd.NA]
        self.df.loc[~self.df['EJECTED'].isin(valid_values_ejected), 'EJECTED'] = np.nan
        
        # Convert types
        self.df['INJ_SEV'] = self.df['INJ_SEV'].astype(object)
        self.df['ROLE'] = self.df['ROLE'].astype(object)
        
        # Create target variable (STATE) based on injury severity
        def create_state(row):
            if pd.isna(row['INJ_SEV']):
                return 'NOT INJURED'
            inj_sev = int(row['INJ_SEV'])
            if inj_sev == 1:
                return 'NOT INJURED'
            elif inj_sev == 8:
                return 'DEAD'
            else:
                return 'INJURED'
        
        self.df['STATE'] = self.df.apply(create_state, axis=1)
        
        # Create additional features
        self.df['VEHICLE_TYPE'] = 'CAR'
        self.df['POSITION'] = np.where(self.df['SEAT_POS'].isin(['1', '2']), 'FRONT', 'BACK') 
        
        # Drop unnecessary columns
        to_delete = ['INJ_SEV', 'SEAT_POS', 'YEAR']
        self.df = self.df.drop(to_delete, axis=1)
        
        logger.info("Data cleaning completed")
    
    def handle_missing_values(self):
        """Handle missing values through imputation."""
        logger.info("Handling missing values...")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            return
        # Identify variable types
        num_var = self.df.select_dtypes(include=[np.number]).columns
        cat_var = self.df.select_dtypes(include=[object]).columns
        
        # Impute numerical variables with median
        num_imputer = SimpleImputer(strategy='median')
        self.df[num_var] = num_imputer.fit_transform(self.df[num_var])
        
        # Impute categorical variables with most frequent value
        cat_imputer = SimpleImputer(strategy='most_frequent')
        self.df[cat_var] = cat_imputer.fit_transform(self.df[cat_var])
        
        missing_count = self.df.isnull().sum().sum()
        logger.info(f"Missing values after imputation: {missing_count}")
    
    def prepare_data(self, sample_size=None):
        """Prepare data for modeling."""
        logger.info(f"Preparing data with sample size: {sample_size}")
        if self.df is None:
            logger.error("Dataframe is not loaded.")
            return None, None
        if sample_size is not None and len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
        X = self.df.drop('STATE', axis=1)
        y = self.df['STATE']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        if self.X_train is None or self.X_test is None:
            logger.error("Train/test split failed.")
            return None, None
        logger.info(f"Training set size: {getattr(self.X_train, 'shape', 'unknown')}")
        logger.info(f"Test set size: {getattr(self.X_test, 'shape', 'unknown')}")
        num_var = ['PARTY_NUM', 'AGE']
        cat_var = ['ROLE', 'EQUIP_1', 'EQUIP_2', 'EJECTED', 'VEHICLE_TYPE', 'POSITION']
        cat_preprocessor = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        num_preprocessor = StandardScaler()
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', cat_preprocessor, cat_var),
                ('num', num_preprocessor, num_var)
            ],
            remainder='drop'
        )
        X_train_processed = self.preprocessor.fit_transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        return X_train_processed, X_test_processed
    
    def train_knn_model(self, X_train_processed, X_test_processed):
        """Train K-Nearest Neighbors model."""
        logger.info("Training K-Nearest Neighbors model...")
        
        # Find optimal k
        k_values = range(1, 21)
        cv_scores = []
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_processed, self.y_train, cv=5)
            cv_scores.append(scores.mean())
        
        best_k = k_values[np.argmax(cv_scores)]
        best_score = max(cv_scores)
        
        logger.info(f"Best k: {best_k} with CV score: {best_score:.4f}")
        
        # Train final model
        best_knn = KNeighborsClassifier(n_neighbors=best_k)
        best_knn.fit(X_train_processed, self.y_train)
        
        # Make predictions
        knn_y_pred = best_knn.predict(X_test_processed)
        knn_accuracy = accuracy_score(self.y_test, knn_y_pred)
        
        logger.info(f"KNN Test Accuracy: {knn_accuracy:.4f}")
        
        # Save results
        self.save_model_results('knn', best_knn, knn_y_pred, knn_accuracy)
        
        return best_knn, knn_y_pred, knn_accuracy
    
    def train_gradient_boosting_model(self, X_train_processed, X_test_processed):
        """Train Gradient Boosting model."""
        logger.info("Training Gradient Boosting model...")
        
        # Find optimal number of estimators
        gb_estimators = range(10, 500, 50)
        gb_scores = []
        
        for n_estimators in gb_estimators:
            gb_clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
            gb_cv_scores = cross_val_score(gb_clf, X_train_processed, self.y_train, cv=5)
            gb_mean_accuracy = gb_cv_scores.mean()
            gb_scores.append(gb_mean_accuracy)
        
        best_estimator = gb_estimators[np.argmax(gb_scores)]
        best_score = max(gb_scores)
        
        logger.info(f"Best estimator: {best_estimator} with CV score: {best_score:.4f}")
        
        # Train final model
        best_gb = GradientBoostingClassifier(n_estimators=best_estimator, random_state=42)
        best_gb.fit(X_train_processed, self.y_train)
        
        # Make predictions
        gb_y_pred = best_gb.predict(X_test_processed)
        gb_accuracy = accuracy_score(self.y_test, gb_y_pred)
        
        logger.info(f"Gradient Boosting Test Accuracy: {gb_accuracy:.4f}")
        
        # Save results
        self.save_model_results('gradient_boosting', best_gb, gb_y_pred, gb_accuracy)
        
        return best_gb, gb_y_pred, gb_accuracy
    
    def train_random_forest_model(self, X_train_processed, X_test_processed):
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        # Find optimal number of estimators
        estimators = range(50, 400, 50)
        precision_scores = []
        
        for n_estimators in estimators:
            rfc = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            scoring = make_scorer(precision_score, average='micro')
            rfc_scores = cross_val_score(rfc, X_train_processed, self.y_train, cv=5, scoring=scoring)
            mean_precision = np.mean(rfc_scores)
            precision_scores.append(mean_precision)
        
        best_n = estimators[np.argmax(precision_scores)]
        best_precision = max(precision_scores)
        
        logger.info(f"Best n_estimators: {best_n} with precision: {best_precision:.4f}")
        
        # Train final model
        best_rfc = RandomForestClassifier(n_estimators=best_n, random_state=42)
        best_rfc.fit(X_train_processed, self.y_train)
        
        # Make predictions
        rfc_y_pred = best_rfc.predict(X_test_processed)
        rfc_accuracy = accuracy_score(self.y_test, rfc_y_pred)
        
        logger.info(f"Random Forest Test Accuracy: {rfc_accuracy:.4f}")
        
        # Save results
        self.save_model_results('random_forest', best_rfc, rfc_y_pred, rfc_accuracy)
        
        return best_rfc, rfc_y_pred, rfc_accuracy
    
    def save_model_results(self, model_name, model, y_pred, accuracy):
        """Save model results and create visualizations."""
        logger.info(f"Saving results for {model_name}...")
        model_path = self.output_dir / f"{model_name}_model.joblib"
        joblib.dump(model, model_path)
        preprocessor_path = self.output_dir / f"{model_name}_preprocessor.joblib"
        joblib.dump(self.preprocessor, preprocessor_path)
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name.replace("_", " ").title()}')
        plt.xlabel('Predicted State')
        plt.ylabel('Actual State')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        report = classification_report(self.y_test, y_pred)
        report_path = self.output_dir / f"{model_name}_classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n\n")
            f.write(str(report))
        logger.info(f"Results saved for {model_name}")
    
    def run_complete_analysis(self):
        """Run the complete injury prediction analysis."""
        logger.info("Starting complete injury prediction analysis...")
        if not self.load_data():
            return False
        self.explore_data()
        self.visualize_accidents_by_year()
        self.visualize_injury_distribution()
        self.clean_data()
        self.handle_missing_values()
        X_train_processed, X_test_processed = self.prepare_data(sample_size=100000)
        if X_train_processed is None or X_test_processed is None:
            logger.error("Data preparation failed.")
            return False
        knn_model, knn_pred, knn_acc = self.train_knn_model(X_train_processed, X_test_processed)
        gb_model, gb_pred, gb_acc = self.train_gradient_boosting_model(X_train_processed, X_test_processed)
        rf_model, rf_pred, rf_acc = self.train_random_forest_model(X_train_processed, X_test_processed)
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"K-Nearest Neighbors Accuracy: {knn_acc:.4f}")
        print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")
        print(f"Random Forest Accuracy: {rf_acc:.4f}")
        print("="*50)
        models = {
            'K-Nearest Neighbors': knn_acc,
            'Gradient Boosting': gb_acc,
            'Random Forest': rf_acc
        }
        best_model = max(models, key=lambda k: models[k])
        best_accuracy = models[best_model]
        print(f"\nBest performing model: {best_model} with accuracy: {best_accuracy:.4f}")
        logger.info("Analysis completed successfully!")
        return True


def main():
    """Main function to run the injury prediction analysis."""
    print("Injury Prediction with Machine Learning")
    print("="*50)
    
    # Create analysis instance
    analyzer = InjuryPredictionML()
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\nAnalysis completed successfully!")
        print(f"Check the '{analyzer.output_dir}' directory for results.")
    else:
        print("\nAnalysis failed. Check the log file for details.")
        sys.exit(1)


if __name__ == "__main__":
    main() 