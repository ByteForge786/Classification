import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import sqlparse
import re
from typing import List, Tuple, Dict, Optional
import joblib
from dataclasses import dataclass
import logging
from pathlib import Path
import csv
from tqdm import tqdm
import torch
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnPrediction:
    table_name: str
    attribute: str
    sensitivity: str
    concept: str
    sensitivity_score: float
    concept_score: float

class SQLParser:
    """Parse SQL CREATE statements and extract column information."""
    
    @staticmethod
    def extract_create_statements(file_path: str) -> List[str]:
        """Extract CREATE statements from a file."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Split by semicolon and filter for CREATE statements
            statements = [stmt.strip() for stmt in content.split(';')
                         if 'CREATE' in stmt.upper()]
            return statements
        except Exception as e:
            logger.error(f"Error reading SQL file: {e}")
            raise

    @staticmethod
    def parse_create_statement(statement: str) -> Tuple[str, List[str]]:
        """Parse a CREATE statement and return table name and columns."""
        try:
            # Parse the SQL statement
            parsed = sqlparse.parse(statement)[0]
            
            # Extract table name
            create_idx = next(i for i, token in enumerate(parsed.tokens) 
                            if token.value.upper().startswith('CREATE'))
            table_idx = next(i for i, token in enumerate(parsed.tokens[create_idx:]) 
                           if token.value.upper().startswith('TABLE'))
            table_name = str(parsed.tokens[create_idx + table_idx + 1]).strip('` ')
            
            # Extract column definitions
            match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if not match:
                raise ValueError(f"No column definitions found in statement: {statement}")
                
            columns_text = match.group(1)
            columns = []
            
            # Split by comma and clean up
            for col in columns_text.split(','):
                col = col.strip()
                if col and not col.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT')):
                    column_name = col.split()[0].strip('` ')
                    columns.append(column_name)
            
            return table_name, columns
        except Exception as e:
            logger.error(f"Error parsing CREATE statement: {e}")
            raise

class AdvancedColumnClassifier:
    """Advanced column classifier using XGBoost and Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.transformer = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
        self.sensitivity_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist',
            max_depth=7,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
        self.concept_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            tree_method='hist',
            max_depth=7,
            n_estimators=200,
            learning_rate=0.1,
            random_state=42
        )
        self.label_encoders = {
            'sensitivity': LabelEncoder(),
            'concept': LabelEncoder()
        }

    def _preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Create combined embeddings using SBERT and TF-IDF."""
        # Get SBERT embeddings
        embeddings = self.transformer.encode(
            texts, 
            show_progress_bar=False, 
            convert_to_numpy=True
        )
        
        # Get TF-IDF features
        tfidf_features = self.tfidf.transform(texts)
        
        # Convert sparse matrix to dense for TF-IDF
        tfidf_dense = tfidf_features.toarray()
        
        # Combine features
        combined_features = np.hstack((embeddings, tfidf_dense))
        return combined_features

    def train(self, master_csv_path: str, test_size: float = 0.2):
        """Train the classifier using the master CSV."""
        try:
            # Load training data
            df = pd.read_csv(master_csv_path)
            logger.info(f"Loaded {len(df)} rows from master CSV")

            # Preprocess features
            X = self._preprocess_text(df['attribute'].tolist())
            
            # Encode labels
            y_sensitivity = self.label_encoders['sensitivity'].fit_transform(df['sensitivity'])
            y_concept = self.label_encoders['concept'].fit_transform(df['concept'])
            
            # Split data
            X_train, X_test, y_sensitivity_train, y_sensitivity_test, \
            y_concept_train, y_concept_test = train_test_split(
                X, y_sensitivity, y_concept, 
                test_size=test_size, 
                random_state=42
            )
            
            # Train sensitivity model
            logger.info("Training sensitivity model...")
            self.sensitivity_model.fit(
                X_train, 
                y_sensitivity_train,
                eval_set=[(X_test, y_sensitivity_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Train concept model
            logger.info("Training concept model...")
            self.concept_model.fit(
                X_train, 
                y_concept_train,
                eval_set=[(X_test, y_concept_test)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Print evaluation metrics
            logger.info("\nSensitivity Model Performance:")
            sensitivity_pred = self.sensitivity_model.predict(X_test)
            print(classification_report(
                y_sensitivity_test, 
                sensitivity_pred,
                target_names=self.label_encoders['sensitivity'].classes_
            ))
            
            logger.info("\nConcept Model Performance:")
            concept_pred = self.concept_model.predict(X_test)
            print(classification_report(
                y_concept_test, 
                concept_pred,
                target_names=self.label_encoders['concept'].classes_
            ))
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def predict(self, columns: List[str]) -> List[Dict]:
        """Predict sensitivity and concept for given columns."""
        try:
            # Preprocess input
            X = self._preprocess_text(columns)
            
            # Get predictions and probabilities
            sensitivity_probs = self.sensitivity_model.predict_proba(X)
            concept_probs = self.concept_model.predict_proba(X)
            
            sensitivity_preds = self.sensitivity_model.predict(X)
            concept_preds = self.concept_model.predict(X)
            
            results = []
            for i, column in enumerate(columns):
                sensitivity = self.label_encoders['sensitivity'].inverse_transform([sensitivity_preds[i]])[0]
                concept = self.label_encoders['concept'].inverse_transform([concept_preds[i]])[0]
                
                sensitivity_score = float(np.max(sensitivity_probs[i]))
                concept_score = float(np.max(concept_probs[i]))
                
                results.append({
                    'attribute': column,
                    'sensitivity': sensitivity,
                    'concept': concept,
                    'sensitivity_score': sensitivity_score,
                    'concept_score': concept_score
                })
            
            return results
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def save_model(self, model_dir: str):
        """Save the trained model and associated components."""
        try:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            
            # Save XGBoost models
            self.sensitivity_model.save_model(f"{model_dir}/sensitivity_model.json")
            self.concept_model.save_model(f"{model_dir}/concept_model.json")
            
            # Save other components
            joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.joblib")
            joblib.dump(self.tfidf, f"{model_dir}/tfidf.joblib")
            
            # Save sentence transformer
            self.transformer.save(f"{model_dir}/sentence_transformer")
            
            logger.info(f"Model saved to {model_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_dir: str) -> 'AdvancedColumnClassifier':
        """Load a trained model."""
        try:
            instance = cls()
            
            # Load XGBoost models
            instance.sensitivity_model.load_model(f"{model_dir}/sensitivity_model.json")
            instance.concept_model.load_model(f"{model_dir}/concept_model.json")
            
            # Load other components
            instance.label_encoders = joblib.load(f"{model_dir}/label_encoders.joblib")
            instance.tfidf = joblib.load(f"{model_dir}/tfidf.joblib")
            
            # Load sentence transformer
            instance.transformer = SentenceTransformer(f"{model_dir}/sentence_transformer")
            
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def process_sql_file(
    sql_file_path: str,
    classifier: AdvancedColumnClassifier,
    output_csv_path: str,
    batch_size: int = 32
):
    """Process SQL file and save predictions to CSV."""
    try:
        # Parse SQL file
        parser = SQLParser()
        create_statements = parser.extract_create_statements(sql_file_path)
        
        # Prepare CSV output
        with open(output_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'table_name', 'attribute', 'sensitivity_predicted',
                'concept_predicted', 'sensitivity_score', 'concept_score'
            ])
            
            # Process CREATE statements in batches
            all_tables = []
            all_columns = []
            
            for statement in tqdm(create_statements, desc="Parsing SQL"):
                table_name, columns = parser.parse_create_statement(statement)
                all_tables.extend([table_name] * len(columns))
                all_columns.extend(columns)
            
            # Process predictions in batches
            for i in tqdm(range(0, len(all_columns), batch_size), desc="Making predictions"):
                batch_columns = all_columns[i:i + batch_size]
                batch_tables = all_tables[i:i + batch_size]
                
                predictions = classifier.predict(batch_columns)
                
                # Write batch predictions to CSV
                for table, pred in zip(batch_tables, predictions):
                    writer.writerow([
                        table,
                        pred['attribute'],
                        pred['sensitivity'],
                        pred['concept'],
                        f"{pred['sensitivity_score']:.4f}",
                        f"{pred['concept_score']:.4f}"
                    ])
        
        logger.info(f"Predictions saved to {output_csv_path}")
        
    except Exception as e:
        logger.error(f"Error processing SQL file: {e}")
        raise

def main():
    """Main execution function."""
    try:
        # Configuration
        master_csv_path = "path/to/master.csv"
        sql_file_path = "path/to/create_statements.sql"
        model_dir = "path/to/model"
        output_csv_path = "path/to/predictions.csv"
        
        # Train or load model
        if Path(f"{model_dir}/sensitivity_model.json").exists():
            classifier = AdvancedColumnClassifier.load_model(model_dir)
            logger.info("Loaded existing model")
        else:
            classifier = AdvancedColumnClassifier()
            classifier.train(master_csv_path)
            classifier.save_model(model_dir)
            logger.info("Trained and saved new model")
        
        # Process SQL file and generate predictions
        process_sql_file(sql_file_path, classifier, output_csv_path)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
