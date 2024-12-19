def analyze_master_data(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Analyze master data to count attributes with same/different sensitivity and concept.
        Uses cleaned attribute names for analysis.
        """
        # Create cleaned attribute column if not exists
        if 'cleaned_attribute' not in df.columns:
            df['cleaned_attribute'] = df['attribute'].apply(clean_attribute_name)
        
        # Group by cleaned attribute to get unique combinations
        sensitivity_concept_by_attr = {}
        
        for _, row in df.iterrows():
            cleaned_attr = row['cleaned_attribute']
            if cleaned_attr not in sensitivity_concept_by_attr:
                sensitivity_concept_by_attr[cleaned_attr] = set()
            
            # Add tuple of (sensitivity, concept) to track unique combinations
            sensitivity_concept_by_attr[cleaned_attr].add((row['sensitivity'], row['concept']))
        
        # Count attributes with same/different variations
        same_count = sum(1 for combinations in sensitivity_concept_by_attr.values() if len(combinations) == 1)
        different_count = sum(1 for combinations in sensitivity_concept_by_attr.values() if len(combinations) > 1)
        
        logger.info(f"Attributes with same sensitivity/concept (after cleaning): {same_count}")
        logger.info(f"Attributes with different sensitivity/concept (after cleaning): {different_count}")
        
        # Print details of attributes with different variations
        if different_count > 0:
            logger.info("\nAttributes with multiple variations:")
            for attr, combinations in sensitivity_concept_by_attr.items():
                if len(combinations) > 1:
                    logger.info(f"\nCleaned Attribute: {attr}")
                    for sens, conc in combinations:
                        logger.info(f"  - Sensitivity: {sens}, Concept: {conc}")
        
        return same_count, different_count



[Previous code remains same until process_batch method, which is replaced with below]

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
                
                # Get all variations from master data for this matched attribute
                matched_variations = self.attribute_variations.get(cleaned_matched, [])
                
                if matched_variations:
                    # Add a prediction for each variation
                    for variation in matched_variations:
                        results.append(ColumnPrediction(
                            table_name=table_name,
                            input_attribute=column,
                            matched_attribute=variation['original_attribute'],
                            sensitivity=variation['sensitivity'],
                            concept=variation['concept'],
                            match_type='semantic',
                            source=variation['source'],
                            domain=variation['domain'],
                            cleaned_input_attribute=cleaned_column,
                            cleaned_matched_attribute=cleaned_matched
                        ))
                else:
                    # If no variations found, use LLM prediction as fallback
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

[Rest of the code remains exactly the same]




def evaluate_predictions(self, master_csv: str, predictions_csv: str) -> Dict:
        """
        Evaluate predictions treating each unique attribute as one record.
        Success criteria:
        - For attributes with same sensitivity/concept in master: Must match exactly
        - For attributes with multiple variations: Must match all variations
        """
        master_df = pd.read_csv(master_csv)
        pred_df = pd.read_csv(predictions_csv)
        
        # Create master attribute dictionary with all variations
        master_dict = {}
        for _, row in master_df.iterrows():
            cleaned_attr = clean_attribute_name(row['attribute'])
            if cleaned_attr not in master_dict:
                master_dict[cleaned_attr] = {
                    'variations': [],
                    'original_names': set()
                }
            master_dict[cleaned_attr]['variations'].append({
                'sensitivity': row['sensitivity'],
                'concept': row['concept'],
                'original_name': row['attribute']
            })
            master_dict[cleaned_attr]['original_names'].add(row['attribute'])
        
        # Analyze each unique attribute
        unique_attributes = set(pred_df['cleaned_input_attribute'])
        total_attributes = len(unique_attributes)
        correct_predictions = 0
        attribute_results = {}
        failures = []
        
        for attr in unique_attributes:
            # Get all predictions for this attribute
            attr_predictions = pred_df[pred_df['cleaned_input_attribute'] == attr]
            input_attr = attr_predictions.iloc[0]['input_attribute']  # Original input name
            
            # Get master variations
            master_info = master_dict.get(attr)
            if not master_info:
                failures.append({
                    'attribute': input_attr,
                    'reason': 'No master data match',
                    'predictions': attr_predictions[['sensitivity', 'concept']].to_dict('records'),
                    'master_variations': []
                })
                continue
            
            master_variations = master_info['variations']
            
            # Check if master has multiple unique sensitivity/concept combinations
            unique_master_variations = set(
                (var['sensitivity'], var['concept']) 
                for var in master_variations
            )
            has_multiple_variations = len(unique_master_variations) > 1
            
            if has_multiple_variations:
                # Must match all variations
                predicted_variations = set(
                    (row['sensitivity'], row['concept']) 
                    for _, row in attr_predictions.iterrows()
                )
                
                all_variations_matched = unique_master_variations == predicted_variations
                
                if all_variations_matched:
                    correct_predictions += 1
                else:
                    failures.append({
                        'attribute': input_attr,
                        'reason': 'Missing variations',
                        'predictions': predicted_variations,
                        'expected': unique_master_variations
                    })
            else:
                # Single variation case - any prediction must match the single variation
                expected_sensitivity = master_variations[0]['sensitivity']
                expected_concept = master_variations[0]['concept']
                
                predictions_correct = all(
                    row['sensitivity'] == expected_sensitivity and 
                    row['concept'] == expected_concept
                    for _, row in attr_predictions.iterrows()
                )
                
                if predictions_correct:
                    correct_predictions += 1
                else:
                    failures.append({
                        'attribute': input_attr,
                        'reason': 'Incorrect prediction',
                        'predictions': attr_predictions[['sensitivity', 'concept']].to_dict('records'),
                        'expected': {
                            'sensitivity': expected_sensitivity,
                            'concept': expected_concept
                        }
                    })
        
        # Calculate accuracy
        accuracy = (correct_predictions / total_attributes * 100) if total_attributes > 0 else 0
        
        # Generate failures CSV
        if failures:
            failures_df = pd.DataFrame([{
                'attribute': f['attribute'],
                'reason': f['reason'],
                'predictions': str(f['predictions']),
                'expected': str(f.get('expected', []))
            } for f in failures])
            failures_df.to_csv('failures.csv', index=False)
        
        evaluation_results = {
            'metrics': {
                'total_unique_attributes': total_attributes,
                'correct_predictions': correct_predictions,
                'failed_predictions': total_attributes - correct_predictions,
                'accuracy': accuracy
            },
            'failures': failures
        }
        
        # Log detailed results
        logger.info("\nEvaluation Results:")
        logger.info(f"Total Unique Attributes: {total_attributes}")
        logger.info(f"Correct Predictions: {correct_predictions}")
        logger.info(f"Failed Predictions: {total_attributes - correct_predictions}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
        
        if failures:
            logger.info("\nFailure Details:")
            for failure in failures:
                logger.info(f"\nAttribute: {failure['attribute']}")
                logger.info(f"Reason: {failure['reason']}")
                if failure['reason'] == 'Missing variations':
                    logger.info(f"Predicted variations: {failure['predictions']}")
                    logger.info(f"Expected variations: {failure['expected']}")
                else:
                    logger.info(f"Predictions made: {failure['predictions']}")
                    logger.info(f"Expected: {failure.get('expected', 'No master data')}")
        
        return evaluation_results
