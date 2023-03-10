# healthinsurance.py

import joblib
import pandas as pd

class HealthGuard:
    """
    A class for predicting health insurance scores.
    """
    def __init__(self, scaler_path: str, ordinal_encoder_path: str, freq_enc_gender_path: str, 
                 freq_enc_region_path: str, freq_enc_policy_path: str, freq_enc_vintage_path: str) -> None:
        """
        Initializes the HealthInsurance class.
        
        :param scaler_path: The path to the scaler object.
        :param ordinal_encoder_path: The path to the ordinal encoder object.
        :param freq_enc_gender_path: The path to the frequency encoder object for gender.
        :param freq_enc_region_path: The path to the frequency encoder object for region.
        :param freq_enc_policy_path: The path to the frequency encoder object for policy.
        :param freq_enc_vintage_path: The path to the frequency encoder object for vintage.
        """
        self.scaler = joblib.load(scaler_path)
        self.ordinal_encoder = joblib.load(ordinal_encoder_path)
        self.freq_enc_gender = joblib.load(freq_enc_gender_path)
        self.freq_enc_region = joblib.load(freq_enc_region_path)
        self.freq_enc_policy = joblib.load(freq_enc_policy_path)
        self.freq_enc_vintage = joblib.load(freq_enc_vintage_path)

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the data by renaming columns.
        
        :param data: The data to clean.
        :returns: The cleaned data.
        """
        cols_new = ['id', 'gender', 'age', 'driving_license', 'region_code',
                    'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
                    'policy_sales_channel', 'vintage']
        data.columns = cols_new
        return data

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers features by dropping the id column.
        
        :param data: The data to engineer features for.
        :returns: The data with features engineered.
        """
        cols_drop = ['id']
        data = data.drop(cols_drop, axis=1)
        return data

def preprocess_data(input_df, scaler, ordinal_encoder, freq_encoders):
    """
    Preprocesses input data for model prediction.
    """
    # Check for missing values
    if input_df.isnull().any().any():
        raise ValueError("Input data contains missing values.")
    
    # Scale numeric features
    numeric_cols = ['age', 'annual_premium']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Encode ordinal features
    input_df['vehicle_age'] = ordinal_encoder.transform(input_df[['vehicle_age']])
    
    # Encode binary feature
    input_df['vehicle_damage'] = input_df['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Encode categorical features using frequency encoding
    for col, enc in freq_encoders.items():
        input_df[col] = input_df[col].map(enc)
    
    # Select subset of features
    selected_cols = ['annual_premium', 'vintage', 'age', 'region_code', 'vehicle_damage',
                     'previously_insured', 'policy_sales_channel']
    input_df = input_df[selected_cols]
    
    return input_df


def predict(model, test_data, scaler, ordinal_encoder, freq_encoders, output_path):
    """
    Predicts the probability of customers buying vehicle insurance and saves the predictions to a file.
    """
    # Preprocess test data
    test_data = preprocess_data(test_data, scaler, ordinal_encoder, freq_encoders)
    
    # Make predictions
    pred = model.predict_proba(test_data)[:, 1]
    
    # Combine predictions with original data and save to file
    output_df = test_data.copy()
    output_df['score'] = pred
    output_df.to_csv(output_path, index=False)
