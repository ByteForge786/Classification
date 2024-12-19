[Previous code remains same until evaluate_predictions method, which is replaced with below]

    def evaluate_predictions(self, master_csv: str, predictions_csv: str) -> Dict:
        """
        Evaluate predictions against master data, treating each attribute as a single entry.
        Handles both single and multiple prediction cases.
        """
        master_df = pd.read_csv(master_csv)
        pred_df = pd.read_csv(predictions_csv)
        
        # Create master data dictionary
        master_dict = {}
        for _, row in master_df.iterrows():
            cleaned_attr = clean_attribute_name(row['attribute'])
            if cleaned_attr not in master_dict:
                master_dict[cleaned_attr] = []
            master_dict[cleaned_attr].append({
                'original_attribute': row['attribute'],
                'sensitivity': row['sensitivity'],
                'concept': row['concept'],
                'source': row.get('source', None),
                'domain': row.get('domain', None)
            })

        # Initialize metrics
        metrics = {
            'total_attributes': len(set(pred_df['input_attribute'])),  # Unique attributes
            'single_pred_attributes': 0,
            'multi_pred_attributes': 0,
            'single_exact_matches': 0,
            'single_semantic_matches': 0,
            'multi_exact_matches': 0,
            'multi_semantic_matches': 0,
            'correct_attributes': 0,
            'failed_attributes': 0
        }

        failures = []
        successes = []

        # Process each unique attribute
        for input_attr, group in pred_df.groupby('input_attribute'):
            cleaned_input = clean_attribute_name(input_attr)
            is_multi_pred = len(group) > 1
            pred_records = group.to_dict('records')
            
            # Update single/multi prediction counts
            if is_multi_pred:
                metrics['multi_pred_attributes'] += 1
            else:
                metrics['single_pred_attributes'] += 1

            # Get master data variations
            master_variations = master_dict.get(cleaned_input, [])
            
            if not master_variations:
                # No master data found
                metrics['failed_attributes'] += 1
                failures.append({
                    'attribute': input_attr,
                    'failure_type': 'No master data match',
                    'prediction_type': 'multiple' if is_multi_pred else 'single',
                    'predictions': pred_records,
                    'master_values': []
                })
                continue

            # Evaluate based on prediction type
            if is_multi_pred:
                # Multiple predictions case
                all_matches_correct = True
                match_results = []

                for pred in pred_records:
                    # Check if this prediction matches any master variation
                    pred_correct = False
                    matched_master = None
                    
                    for master_var in master_variations:
                        if (pred['sensitivity'] == master_var['sensitivity'] and 
                            pred['concept'] == master_var['concept']):
                            pred_correct = True
                            matched_master = master_var
                            break
                    
                    match_results.append({
                        'predicted': pred,
                        'correct': pred_correct,
                        'matched_master': matched_master
                    })
                    
                    if not pred_correct:
                        all_matches_correct = False

                if all_matches_correct:
                    metrics['correct_attributes'] += 1
                    if all(p['match_type'] == 'exact' for p in pred_records):
                        metrics['multi_exact_matches'] += 1
                    else:
                        metrics['multi_semantic_matches'] += 1
                    successes.append({
                        'attribute': input_attr,
                        'prediction_type': 'multiple',
                        'matches': match_results
                    })
                else:
                    metrics['failed_attributes'] += 1
                    failures.append({
                        'attribute': input_attr,
                        'failure_type': 'Incorrect predictions',
                        'prediction_type': 'multiple',
                        'match_results': match_results,
                        'master_values': master_variations
                    })

            else:
                # Single prediction case
                pred = pred_records[0]
                pred_correct = False
                
                for master_var in master_variations:
                    if (pred['sensitivity'] == master_var['sensitivity'] and 
                        pred['concept'] == master_var['concept']):
                        pred_correct = True
                        break

                if pred_correct:
                    metrics['correct_attributes'] += 1
                    if pred['match_type'] == 'exact':
                        metrics['single_exact_matches'] += 1
                    else:
                        metrics['single_semantic_matches'] += 1
                    successes.append({
                        'attribute': input_attr,
                        'prediction_type': 'single',
                        'prediction': pred,
                        'master_match': master_var
                    })
                else:
                    metrics['failed_attributes'] += 1
                    failures.append({
                        'attribute': input_attr,
                        'failure_type': 'Incorrect prediction',
                        'prediction_type': 'single',
                        'prediction': pred,
                        'master_values': master_variations
                    })

        # Generate detailed failure report
        if failures:
            failures_df = pd.DataFrame([{
                'attribute': f['attribute'],
                'failure_type': f['failure_type'],
                'prediction_type': f['prediction_type'],
                'predictions': str(f.get('predictions', f.get('prediction', ''))),
                'master_values': str(f['master_values']),
                'match_results': str(f.get('match_results', ''))
            } for f in failures])
            
            failures_df.to_csv('failures.csv', index=False)

        # Calculate accuracy metrics
        total_attrs = metrics['total_attributes']
        evaluation_results = {
            'Overall Metrics': {
                'Total Unique Attributes': total_attrs,
                'Overall Accuracy': (metrics['correct_attributes'] / total_attrs * 100),
                'Failed Attributes': metrics['failed_attributes']
            },
            'Single Prediction Metrics': {
                'Total Attributes': metrics['single_pred_attributes'],
                'Exact Matches': metrics['single_exact_matches'],
                'Semantic Matches': metrics['single_semantic_matches'],
                'Accuracy': (
                    (metrics['single_exact_matches'] + metrics['single_semantic_matches']) / 
                    metrics['single_pred_attributes'] * 100 if metrics['single_pred_attributes'] > 0 else 0
                )
            },
            'Multiple Prediction Metrics': {
                'Total Attributes': metrics['multi_pred_attributes'],
                'Exact Matches': metrics['multi_exact_matches'],
                'Semantic Matches': metrics['multi_semantic_matches'],
                'Accuracy': (
                    (metrics['multi_exact_matches'] + metrics['multi_semantic_matches']) / 
                    metrics['multi_pred_attributes'] * 100 if metrics['multi_pred_attributes'] > 0 else 0
                )
            },
            'Detailed Failures': failures,
            'Successful Matches': successes
        }

        return evaluation_results

[Previous code remains same until process_sql_file method, where we update the evaluation logging part:]

        # Evaluate predictions
        evaluation = self.evaluate_predictions(config['master_csv'], output_csv)
        
        logger.info("\nEvaluation Results:")
        logger.info("==================")
        
        logger.info("\nOverall Metrics:")
        logger.info(f"Total Unique Attributes: {evaluation['Overall Metrics']['Total Unique Attributes']}")
        logger.info(f"Overall Accuracy: {evaluation['Overall Metrics']['Overall Accuracy']:.2f}%")
        logger.info(f"Failed Attributes: {evaluation['Overall Metrics']['Failed Attributes']}")
        
        logger.info("\nSingle Prediction Attributes:")
        logger.info(f"Total: {evaluation['Single Prediction Metrics']['Total Attributes']}")
        logger.info(f"Exact Matches: {evaluation['Single Prediction Metrics']['Exact Matches']}")
        logger.info(f"Semantic Matches: {evaluation['Single Prediction Metrics']['Semantic Matches']}")
        logger.info(f"Accuracy: {evaluation['Single Prediction Metrics']['Accuracy']:.2f}%")
        
        logger.info("\nMultiple Prediction Attributes:")
        logger.info(f"Total: {evaluation['Multiple Prediction Metrics']['Total Attributes']}")
        logger.info(f"Exact Matches: {evaluation['Multiple Prediction Metrics']['Exact Matches']}")
        logger.info(f"Semantic Matches: {evaluation['Multiple Prediction Metrics']['Semantic Matches']}")
        logger.info(f"Accuracy: {evaluation['Multiple Prediction Metrics']['Accuracy']:.2f}%")
        
        logger.info("\nFailure Summary:")
        if evaluation['Detailed Failures']:
            failure_types = {}
            for failure in evaluation['Detailed Failures']:
                ftype = f"{failure['failure_type']} ({failure['prediction_type']})"
                if ftype not in failure_types:
                    failure_types[ftype] = []
                failure_types[ftype].append(failure['attribute'])
            
            for ftype, attributes in failure_types.items():
                logger.info(f"\n{ftype}:")
                for attr in attributes:
                    logger.info(f"- {attr}")
        else:
            logger.info("No failures detected")








[Previous code remains exactly the same until the evaluate_predictions method, which is replaced with below]

    def evaluate_predictions(self, master_csv: str, predictions_csv: str) -> Dict:
        """
        Evaluate predictions against master data, tracking failures and generating failure report.
        """
        master_df = pd.read_csv(master_csv)
        pred_df = pd.read_csv(predictions_csv)
        
        # Create a dictionary of master data with cleaned attribute names
        master_dict = {}
        for _, row in master_df.iterrows():
            cleaned_attr = clean_attribute_name(row['attribute'])
            if cleaned_attr not in master_dict:
                master_dict[cleaned_attr] = []
            master_dict[cleaned_attr].append({
                'original_attribute': row['attribute'],
                'sensitivity': row['sensitivity'],
                'concept': row['concept'],
                'source': row.get('source', None),
                'domain': row.get('domain', None)
            })

        # Initialize counters and failure tracking
        metrics = {
            'total_predictions': len(pred_df),
            'exact_matches': 0,
            'correct_sensitivity': 0,
            'correct_concept': 0,
            'total_single_pred': 0,
            'total_multi_pred': 0,
            'single_sensitivity_correct': 0,
            'single_concept_correct': 0,
            'multi_sensitivity_correct': 0,
            'multi_concept_correct': 0
        }

        failures = []

        # Group predictions by input_attribute to handle multiple predictions
        pred_groups = pred_df.groupby('input_attribute')
        
        for input_attr, group in pred_groups:
            cleaned_input = clean_attribute_name(input_attr)
            is_multi_pred = len(group) > 1
            
            if is_multi_pred:
                metrics['total_multi_pred'] += 1
            else:
                metrics['total_single_pred'] += 1

            # Get master data variations for this attribute
            master_variations = master_dict.get(cleaned_input, [])
            
            if not master_variations:
                # Track as failure - No master data found
                failures.append({
                    'input_attribute': input_attr,
                    'failure_type': 'No master data match',
                    'predicted_values': group[['matched_attribute', 'sensitivity', 'concept']].to_dict('records'),
                    'master_values': [],
                    'is_multi_pred': is_multi_pred
                })
                continue

            # For exact matches
            if any(row['match_type'] == 'exact' for _, row in group.iterrows()):
                metrics['exact_matches'] += 1
            else:
                # Track as failure - No exact match
                failures.append({
                    'input_attribute': input_attr,
                    'failure_type': 'No exact match',
                    'predicted_values': group[['matched_attribute', 'sensitivity', 'concept']].to_dict('records'),
                    'master_values': master_variations,
                    'is_multi_pred': is_multi_pred
                })

            # Check predictions against all possible master variations
            sensitivity_correct = False
            concept_correct = False
            
            for _, pred_row in group.iterrows():
                # Check if prediction matches any master variation
                for master_var in master_variations:
                    if pred_row['sensitivity'] == master_var['sensitivity']:
                        sensitivity_correct = True
                    if pred_row['concept'] == master_var['concept']:
                        concept_correct = True
                        
                    if sensitivity_correct and concept_correct:
                        break
                        
                if sensitivity_correct and concept_correct:
                    break
            
            # Track failures for sensitivity and concept
            if not sensitivity_correct:
                failures.append({
                    'input_attribute': input_attr,
                    'failure_type': 'Incorrect sensitivity',
                    'predicted_values': group[['matched_attribute', 'sensitivity', 'concept']].to_dict('records'),
                    'master_values': master_variations,
                    'is_multi_pred': is_multi_pred
                })

            if not concept_correct:
                failures.append({
                    'input_attribute': input_attr,
                    'failure_type': 'Incorrect concept',
                    'predicted_values': group[['matched_attribute', 'sensitivity', 'concept']].to_dict('records'),
                    'master_values': master_variations,
                    'is_multi_pred': is_multi_pred
                })

            # Update counters based on prediction type
            if sensitivity_correct:
                metrics['correct_sensitivity'] += 1
                if is_multi_pred:
                    metrics['multi_sensitivity_correct'] += 1
                else:
                    metrics['single_sensitivity_correct'] += 1
            
            if concept_correct:
                metrics['correct_concept'] += 1
                if is_multi_pred:
                    metrics['multi_concept_correct'] += 1
                else:
                    metrics['single_concept_correct'] += 1

        # Generate failures CSV
        if failures:
            failures_df = pd.DataFrame([{
                'input_attribute': f['input_attribute'],
                'failure_type': f['failure_type'],
                'predicted_values': str(f['predicted_values']),
                'master_values': str(f['master_values']),
                'is_multiple_prediction': f['is_multi_pred']
            } for f in failures])
            
            failures_df.to_csv('failures.csv', index=False)
            logger.info(f"\nGenerated failures.csv with {len(failures)} failure records")

        # Calculate percentages
        total = metrics['total_predictions']
        single_total = metrics['total_single_pred'] or 1
        multi_total = metrics['total_multi_pred'] or 1

        evaluation_results = {
            'Overall Metrics': {
                'Total Predictions': total,
                'Total Failures': len(failures),
                'Exact Match Percentage': (metrics['exact_matches'] / total) * 100,
                'Overall Sensitivity Accuracy': (metrics['correct_sensitivity'] / total) * 100,
                'Overall Concept Accuracy': (metrics['correct_concept'] / total) * 100
            },
            'Single Prediction Cases': {
                'Total Count': metrics['total_single_pred'],
                'Sensitivity Accuracy': (metrics['single_sensitivity_correct'] / single_total) * 100,
                'Concept Accuracy': (metrics['single_concept_correct'] / single_total) * 100
            },
            'Multiple Prediction Cases': {
                'Total Count': metrics['total_multi_pred'],
                'Sensitivity Accuracy': (metrics['multi_sensitivity_correct'] / multi_total) * 100,
                'Concept Accuracy': (metrics['multi_concept_correct'] / multi_total) * 100
            },
            'Failures': failures
        }

        return evaluation_results

[Previous code remains the same until process_sql_file method, where we update the evaluation logging part:]

        # Evaluate predictions
        evaluation = self.evaluate_predictions(config['master_csv'], output_csv)
        
        logger.info("\nEvaluation Results:")
        logger.info("==================")
        
        logger.info("\nOverall Metrics:")
        for metric, value in evaluation['Overall Metrics'].items():
            if 'Percentage' in metric or 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info("\nSingle Prediction Cases:")
        for metric, value in evaluation['Single Prediction Cases'].items():
            if 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info("\nMultiple Prediction Cases:")
        for metric, value in evaluation['Multiple Prediction Cases'].items():
            if 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")
        
        logger.info("\nFailure Summary:")
        if evaluation['Failures']:
            failure_types = {}
            for failure in evaluation['Failures']:
                ftype = failure['failure_type']
                if ftype not in failure_types:
                    failure_types[ftype] = []
                failure_types[ftype].append(failure['input_attribute'])
            
            for ftype, attributes in failure_types.items():
                logger.info(f"\n{ftype}:")
                for attr in attributes:
                    logger.info(f"- {attr}")
        else:
            logger.info("No failures detected")







[Previous code remains exactly the same until the evaluate_predictions method, which is replaced with below]

    def evaluate_predictions(self, master_csv: str, predictions_csv: str) -> Dict:
        """
        Evaluate predictions against master data, handling both single and multiple prediction cases.
        Returns detailed accuracy metrics.
        """
        master_df = pd.read_csv(master_csv)
        pred_df = pd.read_csv(predictions_csv)
        
        # Create a dictionary of master data with cleaned attribute names
        master_dict = {}
        for _, row in master_df.iterrows():
            cleaned_attr = clean_attribute_name(row['attribute'])
            if cleaned_attr not in master_dict:
                master_dict[cleaned_attr] = []
            master_dict[cleaned_attr].append({
                'original_attribute': row['attribute'],
                'sensitivity': row['sensitivity'],
                'concept': row['concept'],
                'source': row.get('source', None),
                'domain': row.get('domain', None)
            })

        # Initialize counters
        metrics = {
            'total_predictions': len(pred_df),
            'exact_matches': 0,
            'correct_sensitivity': 0,
            'correct_concept': 0,
            'total_single_pred': 0,
            'total_multi_pred': 0,
            'single_sensitivity_correct': 0,
            'single_concept_correct': 0,
            'multi_sensitivity_correct': 0,
            'multi_concept_correct': 0
        }

        # Group predictions by input_attribute to handle multiple predictions
        pred_groups = pred_df.groupby('input_attribute')
        
        for input_attr, group in pred_groups:
            cleaned_input = clean_attribute_name(input_attr)
            is_multi_pred = len(group) > 1
            
            if is_multi_pred:
                metrics['total_multi_pred'] += 1
            else:
                metrics['total_single_pred'] += 1

            # Get master data variations for this attribute
            master_variations = master_dict.get(cleaned_input, [])
            
            if not master_variations:
                continue  # Skip if no matching master data found
                
            # For exact matches
            if any(row['match_type'] == 'exact' for _, row in group.iterrows()):
                metrics['exact_matches'] += 1

            # Check predictions against all possible master variations
            sensitivity_correct = False
            concept_correct = False
            
            for _, pred_row in group.iterrows():
                # Check if prediction matches any master variation
                for master_var in master_variations:
                    if pred_row['sensitivity'] == master_var['sensitivity']:
                        sensitivity_correct = True
                    if pred_row['concept'] == master_var['concept']:
                        concept_correct = True
                        
                    if sensitivity_correct and concept_correct:
                        break
                        
                if sensitivity_correct and concept_correct:
                    break
            
            # Update counters based on prediction type
            if sensitivity_correct:
                metrics['correct_sensitivity'] += 1
                if is_multi_pred:
                    metrics['multi_sensitivity_correct'] += 1
                else:
                    metrics['single_sensitivity_correct'] += 1
            
            if concept_correct:
                metrics['correct_concept'] += 1
                if is_multi_pred:
                    metrics['multi_concept_correct'] += 1
                else:
                    metrics['single_concept_correct'] += 1

        # Calculate percentages
        total = metrics['total_predictions']
        single_total = metrics['total_single_pred'] or 1  # Avoid division by zero
        multi_total = metrics['total_multi_pred'] or 1    # Avoid division by zero

        evaluation_results = {
            'Overall Metrics': {
                'Total Predictions': total,
                'Exact Match Percentage': (metrics['exact_matches'] / total) * 100,
                'Overall Sensitivity Accuracy': (metrics['correct_sensitivity'] / total) * 100,
                'Overall Concept Accuracy': (metrics['correct_concept'] / total) * 100
            },
            'Single Prediction Cases': {
                'Total Count': metrics['total_single_pred'],
                'Sensitivity Accuracy': (metrics['single_sensitivity_correct'] / single_total) * 100,
                'Concept Accuracy': (metrics['single_concept_correct'] / single_total) * 100
            },
            'Multiple Prediction Cases': {
                'Total Count': metrics['total_multi_pred'],
                'Sensitivity Accuracy': (metrics['multi_sensitivity_correct'] / multi_total) * 100,
                'Concept Accuracy': (metrics['multi_concept_correct'] / multi_total) * 100
            }
        }

        return evaluation_results

[Rest of the code remains exactly the same until process_sql_file method, where we update the evaluation logging part:]

        # Evaluate predictions
        evaluation = self.evaluate_predictions(config['master_csv'], output_csv)
        
        logger.info("\nEvaluation Results:")
        logger.info("==================")
        
        logger.info("\nOverall Metrics:")
        for metric, value in evaluation['Overall Metrics'].items():
            if 'Percentage' in metric or 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info("\nSingle Prediction Cases:")
        for metric, value in evaluation['Single Prediction Cases'].items():
            if 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")
                
        logger.info("\nMultiple Prediction Cases:")
        for metric, value in evaluation['Multiple Prediction Cases'].items():
            if 'Accuracy' in metric:
                logger.info(f"{metric}: {value:.2f}%")
            else:
                logger.info(f"{metric}: {value}")



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

def clean_attribute_name(attr) -> str:
    """Clean attribute name by removing special characters and converting to lowercase."""
    # Convert to string if numeric
    if isinstance(attr, (int, float)):
        attr = str(attr)
    elif attr is None:
        return ''
    
    # Ensure we're working with a string
    try:
        attr = str(attr)
    except Exception as e:
        logger.error(f"Error converting attribute to string: {attr}, type: {type(attr)}, error: {e}")
        return ''
    
    # Remove special characters like [(._ etc
    cleaned = re.sub(r'[\[\]\(\)\.\,\_\{\}\$\#\@\!\?\:\;\&\*\+\-\=]', '', attr)
    # Convert to lowercase and strip whitespace
    return cleaned.lower().strip()

[Rest of the code remains exactly the same as in the previous artifact]



def _preprocess_text(self, texts: List[str], is_training: bool = False) -> np.ndarray:
    """Create combined embeddings using SBERT and TF-IDF."""
    # Get SBERT embeddings
    embeddings = self.transformer.encode(
        texts, 
        show_progress_bar=False, 
        convert_to_numpy=True
    )
    
    # Get TF-IDF features
    if is_training:
        # During training, fit and transform
        tfidf_features = self.tfidf.fit_transform(texts)
    else:
        # During prediction, only transform
        tfidf_features = self.tfidf.transform(texts)
    
    # Convert sparse matrix to dense for TF-IDF
    tfidf_dense = tfidf_features.toarray()
    
    # Combine features
    combined_features = np.hstack((embeddings, tfidf_dense))
    return combined_features

def train(self, master_csv_path: str):
    """Train the classifier using the master CSV."""
    try:
        # Load training data
        df = pd.read_csv(master_csv_path)
        logger.info(f"Loaded {len(df)} rows from master CSV")

        # Preprocess features with is_training=True
        X = self._preprocess_text(df['attribute'].tolist(), is_training=True)
        
        # Rest of the training code remains same...




def train(self, master_csv_path: str):
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
        
        # Train sensitivity model on full dataset
        logger.info("Training sensitivity model...")
        self.sensitivity_model.fit(
            X, 
            y_sensitivity,
            verbose=False
        )
        
        # Train concept model on full dataset
        logger.info("Training concept model...")
        self.concept_model.fit(
            X, 
            y_concept,
            verbose=False
        )
        
        # Print training info
        logger.info(f"Models trained on {len(df)} examples")
        logger.info(f"Sensitivity classes: {self.label_encoders['sensitivity'].classes_}")
        logger.info(f"Concept classes: {self.label_encoders['concept'].classes_}")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise




attribute,sensitivity,concept
user_id,HIGH,CUSTOMER_INFO
email_address,HIGH,CONTACT_INFO
first_name,MEDIUM,PERSONAL_INFO
last_name,MEDIUM,PERSONAL_INFO
phone_number,HIGH,CONTACT_INFO
address_line1,HIGH,ADDRESS
city,MEDIUM,ADDRESS
country,LOW,ADDRESS
birth_date,HIGH,PERSONAL_INFO
age,MEDIUM,DEMOGRAPHIC
gender,MEDIUM,DEMOGRAPHIC
credit_card_number,HIGH,PAYMENT
card_expiry,HIGH,PAYMENT
cvv,HIGH,PAYMENT
account_balance,HIGH,FINANCIAL
transaction_amount,MEDIUM,FINANCIAL
order_id,LOW,TRANSACTION
product_id,LOW,PRODUCT
product_name,LOW,PRODUCT
product_description,LOW,PRODUCT
order_date,LOW,TRANSACTION
shipping_method,LOW,SHIPPING
tracking_number,MEDIUM,SHIPPING
ip_address,HIGH,TECHNICAL
user_agent,MEDIUM,TECHNICAL
password_hash,HIGH,SECURITY
login_attempts,MEDIUM,SECURITY
account_status,LOW,STATUS
created_at,LOW,AUDIT
updated_at,LOW,AUDIT


CREATE TABLE customers (
    cust_id VARCHAR(50),                  -- variation of user_id
    mail_address VARCHAR(100),            -- variation of email_address
    user_firstname VARCHAR(50),           -- variation of first_name
    user_lastname VARCHAR(50),            -- variation of last_name
    mobile_number VARCHAR(20),            -- variation of phone_number
    residential_address TEXT,             -- variation of address_line1
    customer_city VARCHAR(50),            -- variation of city
    cust_country VARCHAR(50),             -- variation of country
    date_of_birth DATE,                   -- variation of birth_date
    customer_age INT,                     -- variation of age
    sex VARCHAR(10),                      -- variation of gender
    credit_card VARCHAR(16)               -- variation of credit_card_number
);

CREATE TABLE payments (
    payment_id VARCHAR(50),
    card_number VARCHAR(16),              -- variation of credit_card_number
    card_valid_until DATE,                -- variation of card_expiry
    security_code VARCHAR(3),             -- variation of cvv
    wallet_balance DECIMAL(10,2),         -- variation of account_balance
    payment_amount DECIMAL(10,2),         -- variation of transaction_amount
    transaction_reference VARCHAR(50)      -- variation of order_id
);

CREATE TABLE products_inventory (
    prod_identifier VARCHAR(50),          -- variation of product_id
    item_name VARCHAR(100),               -- variation of product_name
    item_details TEXT,                    -- variation of product_description
    ship_method VARCHAR(50),              -- variation of shipping_method
    shipment_tracking VARCHAR(50),        -- variation of tracking_number
    visitor_ip VARCHAR(45),               -- variation of ip_address
    browser_info TEXT,                    -- variation of user_agent
    user_pass_hash VARCHAR(128),          -- variation of password_hash
    login_count INT,                      -- variation of login_attempts
    acc_state VARCHAR(20),                -- variation of account_status
    record_created TIMESTAMP,             -- variation of created_at
    last_modified TIMESTAMP               -- variation of updated_at
);

CREATE TABLE users (
    user_identifier VARCHAR(50),
    user_mail VARCHAR(100),
    fname VARCHAR(50),
    lname VARCHAR(50),
    contact_num VARCHAR(20),
    home_address TEXT,
    dob DATE,
    user_gender VARCHAR(10),
    acc_balance DECIMAL(10,2)
);

CREATE TABLE transactions (
    txn_id VARCHAR(50),
    user_ref VARCHAR(50),
    purchase_date TIMESTAMP,
    amount DECIMAL(10,2),
    card_number VARCHAR(16),
    card_expiration DATE,
    security_code VARCHAR(3)
);

CREATE TABLE products (
    item_id VARCHAR(50),
    item_name VARCHAR(100),
    item_desc TEXT,
    price DECIMAL(10,2),
    stock_qty INT,
    category VARCHAR(50)
);

# Configuration
master_csv_path = "master.csv"
sql_file_path = "create_statements.sql"
model_dir = "trained_models"
output_csv_path = "predictions.csv"

# Create and train classifier
classifier = AdvancedColumnClassifier()
classifier.train(master_csv_path)
classifier.save_model(model_dir)

# Process SQL and get predictions
process_sql_file(sql_file_path, classifier, output_csv_path)
