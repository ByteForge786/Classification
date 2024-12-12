from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import sqlparse
import re
from dataclasses import dataclass
import csv
from tqdm import tqdm
import ast
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnPrediction:
    table_name: str
    input_attribute: str
    matched_attribute: str
    sensitivity: str
    concept: str
    match_type: str  # 'exact' or 'semantic'

    def to_csv_row(self) -> List[str]:
        return [
            self.table_name,
            self.input_attribute,
            self.matched_attribute,
            self.sensitivity,
            self.concept,
            self.match_type
        ]

class ColumnClassifier:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.exact_matches = {}
        self.master_attributes = []
        self.master_mappings = {}
        self.index = None
        
    def prepare_master_data(self, master_csv: str):
        """Initialize exact matches and FAISS index from master CSV."""
        logger.info("Loading master data...")
        df = pd.read_csv(master_csv)
        
        # Prepare exact matches dictionary
        for _, row in df.iterrows():
            self.exact_matches[row['attribute'].lower()] = {
                'attribute': row['attribute'],
                'sensitivity': row['sensitivity'],
                'concept': row['concept']
            }
            self.master_mappings[row['attribute']] = {
                'sensitivity': row['sensitivity'],
                'concept': row['concept']
            }
        
        # Prepare FAISS index
        self.master_attributes = df['attribute'].tolist()
        embeddings = self.encoder.encode(
            self.master_attributes,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        logger.info(f"Processed {len(df)} master attributes")

    def get_similar_columns(self, column: str) -> List[str]:
        """Get top 10 similar columns using FAISS."""
        query_embedding = self.encoder.encode([column], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding.astype(np.float32), 10)
        
        similar_columns = []
        for idx in indices[0]:
            attribute = self.master_attributes[idx]
            similar_columns.append({
                'attribute': attribute,
                'sensitivity': self.master_mappings[attribute]['sensitivity'],
                'concept': self.master_mappings[attribute]['concept']
            })
        
        return similar_columns

    def process_batch(self, batch: List[Tuple[str, str]]) -> List[ColumnPrediction]:
        """Process a batch of columns."""
        results = []
        
        for table_name, column in batch:
            # Try exact match first
            exact_match = self.exact_matches.get(column.lower())
            if exact_match:
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute=exact_match['attribute'],
                    sensitivity=exact_match['sensitivity'],
                    concept=exact_match['concept'],
                    match_type='exact'
                ))
                continue
            
            # Get similar columns for LLM context
            similar_columns = self.get_similar_columns(column)
            context = "\n".join([
                f"{col['attribute']}: Sensitivity={col['sensitivity']}, Concept={col['concept']}"
                for col in similar_columns
            ])
            
            # Prepare LLM prompt
            prompt = f"""Given these similar column mappings:
{context}

For this new column: "{column}" from table "{table_name}"
Determine the most appropriate mapping based ONLY on these examples.

Return ONLY a Python dictionary with these keys:
- matched_attribute: The most similar column name from examples
- sensitivity: The sensitivity level (HIGH/MEDIUM/LOW)
- concept: The concept category"""

            # Get LLM prediction
            try:
                llm_result = ast.literal_eval(llm_response(prompt).strip())
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute=llm_result['matched_attribute'],
                    sensitivity=llm_result['sensitivity'],
                    concept=llm_result['concept'],
                    match_type='semantic'
                ))
            except Exception as e:
                logger.error(f"Error processing column {column}: {e}")
                # Add default prediction in case of error
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute='UNKNOWN',
                    sensitivity='UNKNOWN',
                    concept='UNKNOWN',
                    match_type='error'
                ))
        
        return results

    def process_sql_file(self, sql_file: str, output_csv: str):
        """Process SQL file in batches and save results."""
        # Read SQL file
        with open(sql_file, 'r') as f:
            content = f.read()
        
        # Extract CREATE statements
        statements = [stmt.strip() for stmt in content.split(';') 
                     if 'CREATE' in stmt.upper()]
        
        # Collect all columns
        all_columns = []
        for stmt in statements:
            # Extract table name
            table_match = re.search(r'CREATE\s+TABLE\s+([^\s(]+)', stmt, re.IGNORECASE)
            if not table_match:
                continue
            table_name = table_match.group(1).strip('`" ')
            
            # Extract columns
            columns_match = re.search(r'\((.*)\)', stmt, re.DOTALL)
            if not columns_match:
                continue
            
            for col in columns_match.group(1).split(','):
                col = col.strip()
                if col and not any(k in col.upper() for k in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT']):
                    column_name = col.split()[0].strip('`" ')
                    all_columns.append((table_name, column_name))
        
        # Process in batches
        results = []
        for i in tqdm(range(0, len(all_columns), self.batch_size)):
            batch = all_columns[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        
        # Save results
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'table_name',
                'input_attribute',
                'matched_attribute',
                'sensitivity',
                'concept',
                'match_type'
            ])
            
            for result in results:
                writer.writerow(result.to_csv_row())
        
        logger.info(f"Processed {len(results)} columns")

def main():
    """Main execution function."""
    try:
        # Configuration
        config = {
            'master_csv': "master.csv",
            'sql_file': "create_statements.sql",
            'output_csv': "predictions.csv",
            'batch_size': 32
        }
        
        # Initialize and prepare classifier
        classifier = ColumnClassifier(batch_size=config['batch_size'])
        classifier.prepare_master_data(config['master_csv'])
        
        # Process SQL file
        classifier.process_sql_file(config['sql_file'], config['output_csv'])
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
