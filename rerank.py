from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import numpy as np
import faiss
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
import sqlparse
import re
from dataclasses import dataclass
import csv
from tqdm import tqdm
import ast
import sys
from concurrent.futures import ThreadPoolExecutor
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("column_classifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ColumnPrediction:
    table_name: str
    input_attribute: str
    matched_attribute: str
    sensitivity: str
    concept: str
    match_type: str  # 'exact', 'semantic', or 'unknown'

    def to_csv_row(self) -> List[str]:
        return [
            self.table_name,
            self.input_attribute,
            self.matched_attribute,
            self.sensitivity,
            self.concept,
            self.match_type
        ]

class EnhancedColumnClassifier:
    def __init__(
        self,
        batch_size: int = 32,
        initial_retrieval_size: int = 20,
        final_candidates: int = 5,
        use_gpu: bool = torch.cuda.is_available()
    ):
        self.batch_size = batch_size
        self.initial_retrieval_size = initial_retrieval_size
        self.final_candidates = final_candidates
        self.device = 'cuda' if use_gpu else 'cpu'
        
        logger.info(f"Initializing models (device: {self.device})")
        
        # Initialize models
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoder.to(self.device)
        
        self.reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2',
            device=self.device
        )
        
        # Initialize storage
        self.exact_matches = {}
        self.master_attributes = []
        self.master_mappings = {}
        self.index = None

    def prepare_master_data(self, master_csv: str):
        """Initialize indices and mappings from master CSV."""
        try:
            logger.info(f"Loading master data from {master_csv}")
            if not Path(master_csv).exists():
                raise FileNotFoundError(f"Master CSV not found: {master_csv}")
            
            df = pd.read_csv(master_csv)
            
            # Prepare exact matches
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
            
            logger.info("Creating embeddings for master data")
            embeddings = self.encoder.encode(
                self.master_attributes,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=self.device
            )
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype(np.float32))
            
            logger.info(f"Processed {len(df)} master attributes")
            
        except Exception as e:
            logger.error(f"Error preparing master data: {e}")
            raise

    def get_ranked_matches(self, column: str) -> List[Dict]:
        """Get similar columns using FAISS and reranker."""
        try:
            # 1. Get initial candidates using FAISS
            query_embedding = self.encoder.encode(
                [column],
                convert_to_numpy=True,
                device=self.device
            )
            
            _, indices = self.index.search(
                query_embedding.astype(np.float32),
                self.initial_retrieval_size
            )
            
            candidates = [self.master_attributes[idx] for idx in indices[0]]
            
            # 2. Rerank candidates
            pairs = [[column, candidate] for candidate in candidates]
            rerank_scores = self.reranker.predict(pairs)
            
            # 3. Get top matches after reranking
            ranked_results = sorted(
                zip(candidates, rerank_scores),
                key=lambda x: x[1],
                reverse=True
            )[:self.final_candidates]
            
            # 4. Format results
            results = []
            for attribute, _ in ranked_results:
                results.append({
                    'attribute': attribute,
                    **self.master_mappings[attribute]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error in ranking matches for {column}: {e}")
            raise

    def process_batch(self, batch: List[Tuple[str, str]]) -> List[ColumnPrediction]:
        """Process a batch of columns."""
        results = []
        
        for table_name, column in batch:
            try:
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
                
                # Get ranked matches for LLM context
                similar_columns = self.get_ranked_matches(column)
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
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute='UNKNOWN',
                    sensitivity='UNKNOWN',
                    concept='UNKNOWN',
                    match_type='unknown'
                ))
        
        return results

    def process_sql_file(self, sql_file: str, output_csv: str):
        """Process SQL file and save predictions."""
        try:
            logger.info(f"Processing SQL file: {sql_file}")
            
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
                table_match = re.search(
                    r'CREATE\s+TABLE\s+([^\s(]+)',
                    stmt,
                    re.IGNORECASE
                )
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
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
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
            
        except Exception as e:
            logger.error(f"Error processing SQL file: {e}")
            raise

def main():
    """Main execution function."""
    try:
        # Configuration
        config = {
            'master_csv': "master.csv",
            'sql_file': "create_statements.sql",
            'output_csv': "predictions.csv",
            'batch_size': 32,
            'initial_retrieval_size': 20,
            'final_candidates': 5
        }
        
        # Initialize classifier
        classifier = EnhancedColumnClassifier(
            batch_size=config['batch_size'],
            initial_retrieval_size=config['initial_retrieval_size'],
            final_candidates=config['final_candidates']
        )
        
        # Prepare master data
        classifier.prepare_master_data(config['master_csv'])
        
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
