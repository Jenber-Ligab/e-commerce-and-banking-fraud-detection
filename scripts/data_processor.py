import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Modular data processing class for reuse in notebooks"""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_raw_data(self):
        """Load all raw datasets"""
        return (
            pd.read_csv(f'{self.data_path}Fraud_Data.csv'),
            pd.read_csv(f'{self.data_path}IpAddress_to_Country.csv'),
            pd.read_csv(f'{self.data_path}creditcard.csv')
        )

    def process_fraud_data(self, fraud_df, ip_df):
        """Process e-commerce fraud data with docstrings"""
        # Convert timestamps
        fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
        fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
        
        # Calculate time delta between signup and purchase
        fraud_df['signup_to_purchase_hrs'] = (
            (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
        )
        
        # Safely convert IP addresses to integers with validation
        if 'ip_address' in fraud_df.columns:
            # Handle missing values and ensure string type
            fraud_df['ip_address'] = fraud_df['ip_address'].fillna('0.0.0.0').astype(str)
            
            # Convert IP to integer with error handling
            def ip_to_int(ip):
                try:
                    return int(ip.replace('.', ''))
                except (ValueError, AttributeError):
                    return 0  # Return default value for invalid IPs
                    
            fraud_df['ip_address'] = fraud_df['ip_address'].apply(ip_to_int).astype(np.int64)
        
        # Prepare IP lookup data
        ip_df = ip_df.sort_values('lower_bound_ip_address')
        intervals = pd.IntervalIndex.from_arrays(
            ip_df['lower_bound_ip_address'],
            ip_df['upper_bound_ip_address'],
            closed='both'
        )
        
        # Map IPs to countries
        fraud_df['country'] = pd.cut(
            fraud_df['ip_address'],
            bins=intervals,
            labels=ip_df['country']
        )
        
        # Create time-based features
        fraud_df['purchase_hour'] = fraud_df['purchase_time'].dt.hour
        fraud_df['purchase_dayofweek'] = fraud_df['purchase_time'].dt.dayofweek
        
        return fraud_df

    def process_credit_data(self, credit_df):
        """Process credit card data with docstrings""" 
        # Existing processing logic from data_preprocessing.py
        return credit_df  # Return processed DataFrame

    def save_processed_data(self, fraud_df, credit_df):
        """Save processed data to CSV"""
        fraud_df.to_csv(f'{self.data_path}processed_fraud_data.csv', index=False)
        credit_df.to_csv(f'{self.data_path}processed_credit_data.csv', index=False)
