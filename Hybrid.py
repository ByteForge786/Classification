from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime
import os
from dataclasses import dataclass, asdict
import csv
from tqdm import tqdm
import sqlparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
import warnings
from scipy.spatial.distance import cosine

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure logging with both file and console handlers."""
    Path(log_dir).mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"attribute_classifier_{timestamp}.log"
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger("HybridClassifier")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class MatchResult:
    """Data class for storing match results."""
    attribute: str
    matched_to: str
    sensitivity: str
    concept: str
    match_type: str
    confidence: float
    table_name: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_csv_row(self) -> List[str]:
        return [
            str(self.table_name or ''),
            self.attribute,
            self.matched_to,
            self.sensitivity,
            self.concept,
            self.match_type,
            f"{self.confidence:.4f}"
        ]

class ConfigManager:
    """Manage configuration settings."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
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
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

class SQLParser:
    """Parse SQL CREATE statements."""
    
    @staticmethod
    def parse_create_statements(sql_file: str) -> List[Tuple[str, List[str]]]:
        try:
            with open(sql_file, 'r') as f:
                content = f.read()
            
            statements = [stmt.strip() for stmt in content.split(';') 
                         if 'CREATE' in stmt.upper()]
            
            results = []
            for stmt in statements:
                table_match = re.search(r'CREATE\s+TABLE\s+([^\s(]+)', stmt, re.IGNORECASE)
                if not table_match:
                    continue
                    
                table_name = table_match.group(1).strip('`" ')
                
                columns_match = re.search(r'\((.*)\)', stmt, re.DOTALL)
                if not columns_match:
                    continue
                    
                columns = []
                for col in columns_match.group(1).split(','):
                    col = col.strip()
                    if col and not any(k in col.upper() for k in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT']):
                        column_name = col.split()[0].strip('`" ')
                        columns.append(column_name)
                
                results.append((table_name, columns))
            
            return results
        except Exception as e:
            logger.error(f"Error parsing SQL file: {e}")
            raise

class HybridAttributeClassifier:
    """Hybrid approach for attribute classification using exact, semantic, and BERT matching."""
    
    def __init__(
        self,
        semantic_threshold: float = 0.8,
        bert_threshold: float = 0.6,
        model_dir: Optional[str] = None
    ):
        """Initialize the classifier with specified thresholds."""
        self.semantic_threshold = semantic_threshold
        self.bert_threshold = bert_threshold
        
        try:
            # Initialize models
            logger.info("Initializing models...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.bert_model = AutoModel.from_pretrained("bert-base-uncased")
            
            # Initialize storage
            self.exact_mappings: Dict[str, Dict] = {}
            self.semantic_mappings: List[Dict] = []
            self.embeddings_dict: Dict = {}
            
            if model_dir:
                self.load_mappings(model_dir)
                
        except Exception as e:
            logger.error(f"Error initializing classifier: {e}")
            raise

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """Generate BERT embedding for text."""
        try:
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            
        except Exception as e:
            logger.error(f"Error generating BERT embedding: {e}")
            raise

    def train(self, master_csv: str):
        """Train the classifier using master CSV data."""
        try:
            logger.info(f"Loading master CSV: {master_csv}")
            df = pd.read_csv(master_csv)
            
            required_columns = {'attribute', 'sensitivity', 'concept'}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Master CSV missing required columns: {required_columns}")
            
            logger.info(f"Processing {len(df)} attributes...")
            
            # Process each attribute
            for _, row in tqdm(df.iterrows(), total=len(df)):
                attribute = row['attribute'].lower()
                mapping = {
                    'sensitivity': row['sensitivity'],
                    'concept': row['concept']
                }
                
                # Store exact mapping
                self.exact_mappings[attribute] = mapping
                
                # Create and store embeddings
                semantic_embedding = self.sentence_transformer.encode(attribute)
                bert_embedding = self.get_bert_embedding(attribute)
                
                self.semantic_mappings.append({
                    'attribute': attribute,
                    'embedding': semantic_embedding,
                    'sensitivity': row['sensitivity'],
                    'concept': row['concept']
                })
                
                self.embeddings_dict[attribute] = {
                    'sentence_embedding': semantic_embedding,
                    'bert_embedding': bert_embedding,
                    'mapping': mapping
                }
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    def predict(self, column: str, table_name: Optional[str] = None) -> MatchResult:
        """Predict sensitivity and concept for a column using hybrid approach."""
        try:
            column = column.lower()
            
            # 1. Exact Match
            if column in self.exact_mappings:
                mapping = self.exact_mappings[column]
                return MatchResult(
                    attribute=column,
                    matched_to=column,
                    sensitivity=mapping['sensitivity'],
                    concept=mapping['concept'],
                    match_type='exact',
                    confidence=1.0,
                    table_name=table_name
                )
            
            # 2. Semantic Similarity
            column_embedding = self.sentence_transformer.encode(column)
            best_semantic_score = -1
            best_semantic_match = None
            
            for mapping in self.semantic_mappings:
                similarity = 1 - cosine(column_embedding, mapping['embedding'])
                if similarity > best_semantic_score:
                    best_semantic_score = similarity
                    best_semantic_match = mapping
            
            if best_semantic_score > self.semantic_threshold:
                return MatchResult(
                    attribute=column,
                    matched_to=best_semantic_match['attribute'],
                    sensitivity=best_semantic_match['sensitivity'],
                    concept=best_semantic_match['concept'],
                    match_type='semantic',
                    confidence=best_semantic_score,
                    table_name=table_name
                )
            
            # 3. BERT Understanding
            bert_embedding = self.get_bert_embedding(column)
            best_bert_score = -1
            best_bert_match = None
            
            for attr, data in self.embeddings_dict.items():
                similarity = 1 - cosine(bert_embedding, data['bert_embedding'])
                if similarity > best_bert_score:
                    best_bert_score = similarity
                    best_bert_match = {
                        'attribute': attr,
                        **data['mapping']
                    }
            
            if best_bert_score > self.bert_threshold:
                return MatchResult(
                    attribute=column,
                    matched_to=best_bert_match['attribute'],
                    sensitivity=best_bert_match['sensitivity'],
                    concept=best_bert_match['concept'],
                    match_type='bert',
                    confidence=best_bert_score,
                    table_name=table_name
                )
            
            # If no good match found
            logger.warning(f"No good match found for column: {column}")
            return MatchResult(
                attribute=column,
                matched_to='unknown',
                sensitivity='UNKNOWN',
                concept='UNKNOWN',
                match_type='no_match',
                confidence=0.0,
                table_name=table_name
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def process_sql_file(
        self,
        sql_file: str,
        output_csv: str,
        batch_size: int = 32
    ):
        """Process SQL file and generate predictions."""
        try:
            # Parse SQL file
            logger.info(f"Processing SQL file: {sql_file}")
            parser = SQLParser()
            tables_columns = parser.parse_create_statements(sql_file)
            
            # Prepare results
            results = []
            
            # Process each table
            for table_name, columns in tqdm(tables_columns):
                for column in columns:
                    match = self.predict(column, table_name)
                    results.append(match)
            
            # Save results
            self._save_results(results, output_csv)
            logger.info(f"Processed {len(results)} columns from {len(tables_columns)} tables")
            
        except Exception as e:
            logger.error(f"Error processing SQL file: {e}")
            raise

    def _save_results(self, results: List[MatchResult], output_path: str):
        """Save results to CSV file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'table_name',
                    'attribute',
                    'matched_to',
                    'sensitivity',
                    'concept',
                    'match_type',
                    'confidence'
                ])
                
                for result in results:
                    writer.writerow(result.to_csv_row())
                    
            logger.info(f"Saved results to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

    def save_mappings(self, model_dir: str):
        """Save all mappings and embeddings."""
        try:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save mappings
            np.save(str(model_dir / 'exact_mappings.npy'), self.exact_mappings)
            np.save(str(model_dir / 'semantic_mappings.npy'), self.semantic_mappings)
            np.save(str(model_dir / 'embeddings_dict.npy'), self.embeddings_dict)
            
            # Save config
            config = {
                'semantic_threshold': self.semantic_threshold,
                'bert_threshold': self.bert_threshold
            }
            ConfigManager.save_config(config, str(model_dir / 'config.json'))
            
            logger.info(f"Saved mappings to {model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
            raise

    def load_mappings(self, model_dir: str):
        """Load saved mappings and embeddings."""
        try:
            model_dir = Path(model_dir)
            
            # Load config
            config = ConfigManager.load_config(str(model_dir / 'config.json'))
            self.semantic_threshold = config['semantic_threshold']
            self.bert_threshold = config['bert_threshold']
            
            # Load mappings
            self.exact_mappings = np.load(
                str(model_dir / 'exact_mappings.npy'),
                allow_pickle=True
            ).item()
            self.semantic_mappings = np.load(
                str(model_dir / 'semantic_mappings.npy'),
                allow_pickle=True
            )
            self.embeddings_dict = np.load(
                str(model_dir / 'embeddings_dict.npy'),
                allow_pickle=True
            ).item()
            
            logger.info(f"Loaded mappings from {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            raise

def main():
    """Main execution function."""
    try:
        # Configuration
        config = {
            'master_csv': "data/master.csv",
            'sql_file': "data/create_statements.sql",
            'model_dir': "models/attribute_classifier",
            'output_csv': "output/predictions.csv",
            'semantic_threshold': 0.8,
            'bert_threshold': 0.6
        }
        
        # Create or load classifier
        if Path(config['model_dir']).exists():
            classifier = HybridAttributeClassifier(
                semantic_threshold=config['semantic_threshold'],
                bert_threshold=config['bert_threshold'],
                model_dir=config['model_dir']
            )
            logger.info("Loaded existing model")
        else:
            classifier = HybridAttributeClassifier(
                semantic_threshold=config['semantic_threshold'],
                bert_threshold=config['bert_threshold']
            )
            classifier.train(config['master_csv'])
            classifier.save_mappings(config['model_dir'])
            logger.info("Created and saved new model")
        
        # Process SQL file
        classifier.process_sql_file(
            config['sql_file'],
            config['output_csv']
        )
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
