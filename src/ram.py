import pandas as pd
import numpy as np

def low_ram(df: pd.DataFrame, target_format: str = 'csv') -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype

        if 'int' in str(col_type):
            if target_format == 'csv':
                df[col] = df[col].astype(np.int8)
            else:
                df[col] = df[col].astype(np.int16) 
            
        elif 'float' in str(col_type):
            df[col] = df[col].astype(np.float32)
            
    return df