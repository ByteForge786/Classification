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

def clean_attribute_name(attr: str) -> str:
    """Clean attribute name by removing special characters and converting to lowercase."""
    # Remove special characters like [(._ etc
    cleaned = re.sub(r'[\[\]\(\)\.\,\_\{\}\$\#\@\!\?\:\;\&\*\+\-\=]', '', attr)
    # Convert to lowercase and strip whitespace
    return cleaned.lower().strip()

@dataclass
class ColumnPrediction:
    table_name: str
    input_attribute: str
    matched_attribute: str
    sensitivity: str
    concept: str
    match_type: str  # 'exact' or 'semantic'
    source: str = None
    domain: str = None
    original_matched_attribute: str = None
    cleaned_input_attribute: str = None
    cleaned_matched_attribute: str = None

    def to_csv_row(self) -> List[str]:
        return [
            self.table_name,
            self.input_attribute,  # Original input attribute
            self.matched_attribute,  # Original matched attribute
            self.sensitivity,
            self.concept,
            self.match_type,
            self.source or 'None',
            self.domain or 'None',
            self.cleaned_input_attribute,  # Add cleaned versions for verification
            self.cleaned_matched_attribute
        ]

class ColumnClassifier:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.exact_matches = {}
        self.master_attributes = []
        self.master_mappings = {}
        self.index = None
        self.attribute_variations = {}
        
    def analyze_master_data(self, df: pd.DataFrame) -> Tuple[int, int]:
        """Analyze master data to count attributes with same/different sensitivity and concept."""
        df['cleaned_attribute'] = df['attribute'].apply(clean_attribute_name)
        attribute_groups = df.groupby('cleaned_attribute').agg({
            'sensitivity': lambda x: len(set(x)),
            'concept': lambda x: len(set(x))
        })
        
        same_count = len(attribute_groups[
            (attribute_groups['sensitivity'] == 1) & 
            (attribute_groups['concept'] == 1)
        ])
        different_count = len(attribute_groups) - same_count
        
        logger.info(f"Attributes with same sensitivity/concept: {same_count}")
        logger.info(f"Attributes with different sensitivity/concept: {different_count}")
        
        return same_count, different_count
        
    def prepare_master_data(self, master_csv: str):
        """Initialize exact matches and FAISS index from master CSV."""
        logger.info("Loading master data...")
        df = pd.read_csv(master_csv)
        
        # Add cleaned attribute column
        df['cleaned_attribute'] = df['attribute'].apply(clean_attribute_name)
        
        # Analyze master data
        self.analyze_master_data(df)
        
        # Group by cleaned attribute name to handle variations
        for cleaned_attr, group in df.groupby('cleaned_attribute'):
            variations = []
            
            for _, row in group.iterrows():
                variation = {
                    'original_attribute': row['attribute'],
                    'cleaned_attribute': cleaned_attr,
                    'sensitivity': row['sensitivity'],
                    'concept': row['concept'],
                    'source': row.get('source', None),
                    'domain': row.get('domain', None)
                }
                variations.append(variation)
            
            self.attribute_variations[cleaned_attr] = variations
            self.exact_matches[cleaned_attr] = variations
        
        # Prepare FAISS index with unique cleaned attributes
        self.master_attributes = list(set(df['cleaned_attribute'].tolist()))
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
        logger.info(f"Unique cleaned attributes: {len(self.master_attributes)}")

    def get_similar_columns(self, column: str) -> List[Dict]:
        """Get top 10 similar columns using FAISS."""
        cleaned_column = clean_attribute_name(column)
        query_embedding = self.encoder.encode([cleaned_column], convert_to_numpy=True)
        _, indices = self.index.search(query_embedding.astype(np.float32), 10)
        
        similar_columns = []
        for idx in indices[0]:
            attribute = self.master_attributes[idx]
            variations = self.attribute_variations[attribute]
            for var in variations:
                similar_columns.append(var)
        
        return similar_columns

    def process_batch(self, batch: List[Tuple[str, str]]) -> List[ColumnPrediction]:
        """Process a batch of columns."""
        results = []
        
        for table_name, column in batch:
            cleaned_column = clean_attribute_name(column)
            
            # Try exact match first with cleaned attribute
            exact_matches = self.exact_matches.get(cleaned_column, [])
            if exact_matches:
                for match in exact_matches:
                    results.append(ColumnPrediction(
                        table_name=table_name,
                        input_attribute=column,
                        matched_attribute=match['original_attribute'],
                        sensitivity=match['sensitivity'],
                        concept=match['concept'],
                        match_type='exact',
                        source=match['source'],
                        domain=match['domain'],
                        original_matched_attribute=match['original_attribute'],
                        cleaned_input_attribute=cleaned_column,
                        cleaned_matched_attribute=match['cleaned_attribute']
                    ))
                continue
            
            # Get similar columns for LLM context
            similar_columns = self.get_similar_columns(cleaned_column)
            context = "\n".join([
                f"{col['original_attribute']} (cleaned: {col['cleaned_attribute']}): "
                f"Sensitivity={col['sensitivity']}, "
                f"Concept={col['concept']}, Source={col.get('source', 'None')}, "
                f"Domain={col.get('domain', 'None')}"
                for col in similar_columns
            ])
            
            # Prepare LLM prompt
            prompt = f"""Given these similar column mappings:
{context}

For this new column: "{column}" (cleaned: "{cleaned_column}") from table "{table_name}"
Determine the most appropriate mapping based ONLY on these examples.

Return ONLY a Python dictionary with these keys:
- matched_attribute: The most similar column name from examples
- sensitivity: The sensitivity level (HIGH/MEDIUM/LOW)
- concept: The concept category
- source: The source if available
- domain: The domain if available"""

            try:
                llm_result = ast.literal_eval(llm_response(prompt).strip())
                matched_attr = llm_result['matched_attribute']
                cleaned_matched = clean_attribute_name(matched_attr)
                
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute=matched_attr,
                    sensitivity=llm_result['sensitivity'],
                    concept=llm_result['concept'],
                    match_type='semantic',
                    source=llm_result.get('source'),
                    domain=llm_result.get('domain'),
                    cleaned_input_attribute=cleaned_column,
                    cleaned_matched_attribute=cleaned_matched
                ))
            except Exception as e:
                logger.error(f"Error processing column {column} (cleaned: {cleaned_column}): {e}")
                results.append(ColumnPrediction(
                    table_name=table_name,
                    input_attribute=column,
                    matched_attribute='UNKNOWN',
                    sensitivity='UNKNOWN',
                    concept='UNKNOWN',
                    match_type='error',
                    source=None,
                    domain=None,
                    cleaned_input_attribute=cleaned_column,
                    cleaned_matched_attribute='unknown'
                ))
        
        return results

    def evaluate_predictions(self, master_csv: str, predictions_csv: str) -> Dict:
        """Evaluate predictions against master data."""
        master_df = pd.read_csv(master_csv)
        pred_df = pd.read_csv(predictions_csv)
        
        # Create dictionaries for quick lookup using cleaned attributes
        master_dict = {}
        for _, row in master_df.iterrows():
            cleaned_attr = clean_attribute_name(row['attribute'])
            if cleaned_attr not in master_dict:
                master_dict[cleaned_attr] = []
            master_dict[cleaned_attr].append({
                'sensitivity': row['sensitivity'],
                'concept': row['concept']
            })
        
        # Calculate metrics
        total_unique_attrs = len(set(pred_df['input_attribute']))
        exact_matches = len(pred_df[pred_df['match_type'] == 'exact'])
        correct_sensitivity = 0
        correct_concept = 0
        
        for _, row in pred_df.iterrows():
            cleaned_pred = clean_attribute_name(row['matched_attribute'])
            if cleaned_pred in master_dict:
                master_variations = master_dict[cleaned_pred]
                for var in master_variations:
                    if var['sensitivity'] == row['sensitivity']:
                        correct_sensitivity += 1
                        break
                for var in master_variations:
                    if var['concept'] == row['concept']:
                        correct_concept += 1
                        break
        
        return {
            'total_unique_attributes': total_unique_attrs,
            'total_predictions': len(pred_df),
            'exact_matches': exact_matches,
            'exact_match_percentage': (exact_matches / len(pred_df)) * 100,
            'sensitivity_accuracy': (correct_sensitivity / len(pred_df)) * 100,
            'concept_accuracy': (correct_concept / len(pred_df)) * 100
        }

    def process_sql_file(self, sql_file: str, output_csv: str):
        """Process SQL file in batches and save results."""
        with open(sql_file, 'r') as f:
            content = f.read()
        
        statements = [stmt.strip() for stmt in content.split(';') 
                     if 'CREATE' in stmt.upper()]
        
        all_columns = []
        for stmt in statements:
            table_match = re.search(r'CREATE\s+TABLE\s+([^\s(]+)', stmt, re.IGNORECASE)
            if not table_match:
                continue
            table_name = table_match.group(1).strip('`" ')
            
            columns_match = re.search(r'\((.*)\)', stmt, re.DOTALL)
            if not columns_match:
                continue
            
            for col in columns_match.group(1).split(','):
                col = col.strip()
                if col and not any(k in col.upper() for k in ['PRIMARY KEY', 'FOREIGN KEY', 'CONSTRAINT']):
                    column_name = col.split()[0].strip('`" ')
                    all_columns.append((table_name, column_name))
        
        results = []
        for i in tqdm(range(0, len(all_columns), self.batch_size)):
            batch = all_columns[i:i + self.batch_size]
            batch_results = self.process_batch(batch)
            results.extend(batch_results)
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'table_name',
                'input_attribute',
                'matched_attribute',
                'sensitivity',
                'concept',
                'match_type',
                'source',
                'domain',
                'cleaned_input_attribute',
                'cleaned_matched_attribute'
            ])
            
            for result in results:
                writer.writerow(result.to_csv_row())
        
        logger.info(f"Processed {len(results)} columns")
        
        evaluation = self.evaluate_predictions(config['master_csv'], output_csv)
        logger.info("\nEvaluation Results:")
        for metric, value in evaluation.items():
            if 'percentage' in metric.lower() or 'accuracy' in metric.lower():
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")

def main():
    """Main execution function."""
    try:
        config = {
            'master_csv': "master.csv",
            'sql_file': "create_statements.sql",
            'output_csv': "predictions.csv",
            'batch_size': 32
        }
        
        classifier = ColumnClassifier(batch_size=config['batch_size'])
        classifier.prepare_master_data(config['master_csv'])
        classifier.process_sql_file(config['sql_file'], config['output_csv'])
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
