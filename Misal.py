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
