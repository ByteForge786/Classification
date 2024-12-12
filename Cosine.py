import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlparse
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import os
from dataclasses import dataclass, asdict
import csv
from tqdm import tqdm
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Set up logging configuration"""
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"attribute_classifier_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Set up logger
    logger = logging.getLogger("AttributeClassifier")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class ColumnMapping:
    """Data class for storing column mappings"""
    attribute: str
    sensitivity: str
    concept: str
    embeddings: np.ndarray
    
    def to_dict(self) -> Dict:
        """Convert to dictionary excluding embeddings"""
        return {
            'attribute': self.attribute,
            'sensitivity': self.sensitivity,
            'concept': self.concept
        }

@dataclass
class MatchResult:
    """Data class for storing match results"""
    table_name: str
    input_attribute: str
    matched_attribute: str
    sensitivity: str
    concept: str
    confidence_score: float
    
    def to_csv_row(self) -> List[str]:
        """Convert to CSV row"""
        return [
            self.table_name,
            self.input_attribute,
            self.matched_attribute,
            self.sensitivity,
            self.concept,
            f"{self.confidence_score:.4f}"
        ]

class ConfigManager:
    """Manage configuration settings"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    @staticmethod
    def save_config(config: Dict, config_path: str):
        """Save configuration to JSON file"""
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

class SQLParser:
    """Parse SQL CREATE statements"""
    
    @staticmethod
    def extract_create_statements(sql_content: str) -> List[str]:
        """Extract CREATE TABLE statements from SQL content"""
        try:
            statements = sqlparse.split(sql_content)
            create_statements = [
                stmt for stmt in statements
                if stmt.strip().upper().startswith('CREATE TABLE')
            ]
            return create_statements
        except Exception as e:
            logger.error(f"Error extracting CREATE statements: {e}")
            raise

    @staticmethod
    def parse_create_statement(statement: str) -> Tuple[str, List[str]]:
        """Parse CREATE statement to extract table name and columns"""
        try:
            # Parse statement
            parsed = sqlparse.parse(statement)[0]
            
            # Extract table name
            table_name = ''
            for token in parsed.tokens:
                if token.ttype is None and isinstance(token, sqlparse.sql.Identifier):
                    table_name = token.get_name()
                    break
            
            if not table_name:
                raise ValueError(f"Could not extract table name from: {statement}")
            
            # Extract columns
            columns_match = re.search(r'\((.*)\)', statement, re.DOTALL)
            if not columns_match:
                raise ValueError(f"No column definitions found in: {statement}")
                
            columns_text = columns_match.group(1)
            columns = []
            
            for col in columns_text.split(','):
                col = col.strip()
                if col and not any(k in col.upper() for k in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT']):
                    column_name = col.split()[0].strip('` "')
                    columns.append(column_name)
            
            return table_name, columns
            
        except Exception as e:
            logger.error(f"Error parsing CREATE statement: {e}")
            raise

class AttributeClassifier:
    """Main class for attribute classification"""
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        similarity_threshold: float = 0.7,
        batch_size: int = 32
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.mappings: List[ColumnMapping] = []

    def train(self, master_csv_path: str):
        """Process master CSV and create embeddings"""
        try:
            # Validate file
            if not Path(master_csv_path).exists():
                raise FileNotFoundError(f"Master CSV not found: {master_csv_path}")
            
            # Load data
            df = pd.read_csv(master_csv_path)
            required_columns = {'attribute', 'sensitivity', 'concept'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Master CSV missing required columns: {required_columns}")
            
            logger.info(f"Processing {len(df)} attributes from master CSV")
            
            # Create embeddings
            attributes = df['attribute'].tolist()
            embeddings = self.model.encode(
                attributes,
                batch_size=self.batch_size,
                show_progress_bar=True
            )
            
            # Create mappings
            self.mappings = [
                ColumnMapping(
                    attribute=row['attribute'],
                    sensitivity=row['sensitivity'],
                    concept=row['concept'],
                    embeddings=embeddings[idx]
                )
                for idx, row in df.iterrows()
            ]
            
            logger.info("Successfully created attribute mappings")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            raise

    def find_closest_match(self, column: str) -> Optional[MatchResult]:
        """Find closest matching attribute"""
        try:
            # Get embedding for input column
            column_embedding = self.model.encode(column)
            
            # Calculate similarities
            similarities = [
                np.dot(column_embedding, m.embeddings) / 
                (np.linalg.norm(column_embedding) * np.linalg.norm(m.embeddings))
                for m in self.mappings
            ]
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            
            if best_score < self.similarity_threshold:
                logger.warning(f"No good match found for column: {column}")
                return None
            
            best_match = self.mappings[best_idx]
            return best_match, best_score
            
        except Exception as e:
            logger.error(f"Error finding match for {column}: {e}")
            raise

    def process_sql_file(
        self,
        sql_file_path: str,
        output_csv_path: str
    ):
        """Process SQL file and generate predictions"""
        try:
            # Read SQL file
            with open(sql_file_path, 'r') as f:
                sql_content = f.read()
            
            # Parse SQL
            parser = SQLParser()
            create_statements = parser.extract_create_statements(sql_content)
            
            # Process statements and collect results
            results: List[MatchResult] = []
            
            for statement in tqdm(create_statements, desc="Processing tables"):
                table_name, columns = parser.parse_create_statement(statement)
                
                for column in columns:
                    match = self.find_closest_match(column)
                    if match:
                        best_match, score = match
                        results.append(
                            MatchResult(
                                table_name=table_name,
                                input_attribute=column,
                                matched_attribute=best_match.attribute,
                                sensitivity=best_match.sensitivity,
                                concept=best_match.concept,
                                confidence_score=score
                            )
                        )
            
            # Save results
            self._save_results(results, output_csv_path)
            logger.info(f"Processed {len(create_statements)} tables, {len(results)} columns")
            
        except Exception as e:
            logger.error(f"Error processing SQL file: {e}")
            raise

    def _save_results(self, results: List[MatchResult], output_path: str):
        """Save results to CSV"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'table_name',
                    'input_attribute',
                    'matched_attribute',
                    'sensitivity',
                    'concept',
                    'confidence_score'
                ])
                
                for result in results:
                    writer.writerow(result.to_csv_row())
                    
            logger.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def save_model(self, model_dir: str):
        """Save model and mappings"""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save sentence transformer
            self.model.save(str(model_dir / 'sentence_transformer'))
            
            # Save mappings
            mappings_data = {
                'attributes': [m.attribute for m in self.mappings],
                'sensitivities': [m.sensitivity for m in self.mappings],
                'concepts': [m.concept for m in self.mappings],
                'embeddings': [m.embeddings for m in self.mappings]
            }
            np.save(str(model_dir / 'mappings.npy'), mappings_data)
            
            # Save config
            config = {
                'similarity_threshold': self.similarity_threshold,
                'batch_size': self.batch_size
            }
            ConfigManager.save_config(config, str(model_dir / 'config.json'))
            
            logger.info(f"Saved model and mappings to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    @classmethod
    def load_model(cls, model_dir: str) -> 'AttributeClassifier':
        """Load saved model and mappings"""
        try:
            model_dir = Path(model_dir)
            
            # Load config
            config = ConfigManager.load_config(str(model_dir / 'config.json'))
            
            # Create instance
            instance = cls(
                similarity_threshold=config['similarity_threshold'],
                batch_size=config['batch_size']
            )
            
            # Load sentence transformer
            instance.model = SentenceTransformer(str(model_dir / 'sentence_transformer'))
            
            # Load mappings
            mappings_data = np.load(str(model_dir / 'mappings.npy'), allow_pickle=True).item()
            instance.mappings = [
                ColumnMapping(a, s, c, e)
                for a, s, c, e in zip(
                    mappings_data['attributes'],
                    mappings_data['sensitivities'],
                    mappings_data['concepts'],
                    mappings_data['embeddings']
                )
            ]
            
            logger.info(f"Loaded model from {model_dir}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

def main():
    """Main execution function"""
    try:
        # Configuration
        config = {
            'master_csv': "data/master.csv",
            'sql_file': "data/create_statements.sql",
            'model_dir': "models/attribute_classifier",
            'output_csv': "output/predictions.csv",
            'similarity_threshold': 0.7,
            'batch_size': 32
        }
        
        # Create or load classifier
        if Path(config['model_dir']).exists():
            classifier = AttributeClassifier.load_model(config['model_dir'])
            logger.info("Loaded existing model")
        else:
            classifier = AttributeClassifier(
                similarity_threshold=config['similarity_threshold'],
                batch_size=config['batch_size']
            )
            classifier.train(config['master_csv'])
            classifier.save_model(config['model_dir'])
            logger.info("Created and saved new model")
        
        # Process SQL file
        classifier.process_sql_file(config['sql_file'], config['output_csv'])
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
