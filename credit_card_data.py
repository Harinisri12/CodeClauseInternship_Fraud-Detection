import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate synthetic data
np.random.seed(42)

# Generate normal transactions
normal_transactions = pd.DataFrame({
    'Time': np.arange(0, 1000),
    'Amount': np.random.uniform(1, 500, 1000),
    'Class': 0  # 0 represents normal transactions
})

# Generate fraudulent transactions
fraudulent_transactions = pd.DataFrame({
    'Time': np.arange(1000, 1100),
    'Amount': np.random.uniform(1000, 5000, 100),
    'Class': 1  # 1 represents fraudulent transactions
})

# Concatenate normal and fraudulent transactions
credit_card_data = pd.concat([normal_transactions, fraudulent_transactions], ignore_index=True)

# Shuffle the dataset
credit_card_data = credit_card_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the dataset to a CSV file
credit_card_data.to_csv('credit_card_data.csv', index=False)

# Display the first few rows of the generated dataset
print(credit_card_data.head())
