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
