from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict
import sqlparse
import re
from pathlib import Path
import csv
from tqdm import tqdm
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ColumnMapping:
    attribute: str
    sensitivity: str
    concept: str
    embeddings: np.ndarray

class AttributeMapper:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.mappings: List[ColumnMapping] = []
        
    def train(self, master_csv_path: str):
        """Memorize mappings from master CSV"""
        df = pd.read_csv(master_csv_path)
        logger.info(f"Loading {len(df)} attributes from master CSV")
        
        # Create embeddings for all attributes
        attributes = df['attribute'].tolist()
        embeddings = self.model.encode(attributes)
        
        # Store mappings
        for idx, row in df.iterrows():
            self.mappings.append(
                ColumnMapping(
                    attribute=row['attribute'],
                    sensitivity=row['sensitivity'],
                    concept=row['concept'],
                    embeddings=embeddings[idx]
                )
            )
        
        logger.info("Mappings created and stored")
    
    def find_closest_match(self, column: str, threshold: float = 0.7) -> Dict:
        """Find closest matching attribute and return its mapping"""
        # Get embedding for input column
        column_embedding = self.model.encode(column)
        
        # Calculate similarities with all stored mappings
        similarities = [
            np.dot(column_embedding, m.embeddings) / 
            (np.linalg.norm(column_embedding) * np.linalg.norm(m.embeddings))
            for m in self.mappings
        ]
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        best_match = self.mappings[best_idx]
        
        return {
            'input_attribute': column,
            'matched_attribute': best_match.attribute,
            'sensitivity': best_match.sensitivity,
            'concept': best_match.concept,
            'confidence_score': float(best_score)
        }
    
    def save_mappings(self, path: str):
        """Save the model and mappings"""
        data = {
            'attributes': [m.attribute for m in self.mappings],
            'sensitivities': [m.sensitivity for m in self.mappings],
            'concepts': [m.concept for m in self.mappings],
            'embeddings': [m.embeddings for m in self.mappings]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data)
        
    @classmethod
    def load_mappings(cls, path: str) -> 'AttributeMapper':
        """Load saved mappings"""
        data = np.load(path, allow_pickle=True).item()
        instance = cls()
        instance.mappings = [
            ColumnMapping(a, s, c, e)
            for a, s, c, e in zip(
                data['attributes'],
                data['sensitivities'],
                data['concepts'],
                data['embeddings']
            )
        ]
        return instance

def process_sql_file(
    sql_file_path: str,
    mapper: AttributeMapper,
    output_csv_path: str
):
    """Process SQL file and generate predictions"""
    # Read and parse SQL
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    # Split into CREATE statements
    statements = [s.strip() for s in sql_content.split(';') if 'CREATE' in s.upper()]
    
    # Process each statement
    results = []
    for stmt in statements:
        # Extract table name
        table_match = re.search(r'CREATE\s+TABLE\s+(\w+)', stmt, re.IGNORECASE)
        if not table_match:
            continue
        table_name = table_match.group(1)
        
        # Extract columns
        columns_match = re.search(r'\((.*)\)', stmt, re.DOTALL)
        if not columns_match:
            continue
            
        columns_text = columns_match.group(1)
        columns = [
            col.split()[0].strip('`, ')
            for col in columns_text.split(',')
            if col.strip() and not any(k in col.upper() for k in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT'])
        ]
        
        # Find matches for each column
        for col in columns:
            match = mapper.find_closest_match(col)
            results.append({
                'table_name': table_name,
                'input_attribute': col,
                'matched_attribute': match['matched_attribute'],
                'sensitivity': match['sensitivity'],
                'concept': match['concept'],
                'confidence_score': match['confidence_score']
            })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved predictions to {output_csv_path}")

def main():
    # Paths
    master_csv = "master.csv"
    sql_file = "create_statements.sql"
    output_file = "predictions.csv"
    mappings_file = "mappings.npy"
    
    # Create or load mapper
    if Path(mappings_file).exists():
        mapper = AttributeMapper.load_mappings(mappings_file)
        logger.info("Loaded existing mappings")
    else:
        mapper = AttributeMapper()
        mapper.train(master_csv)
        mapper.save_mappings(mappings_file)
        logger.info("Created and saved new mappings")
    
    # Process SQL file
    process_sql_file(sql_file, mapper, output_file)

if __name__ == "__main__":
    main()
