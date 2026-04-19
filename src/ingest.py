#!/usr/bin/env python3
import sys
import os
import pandas as pd
import joblib
from pathlib import Path

def ingest_data(file_path):
    try:
        if not file_path:
            print("Usage: python src/ingest.py <path_to_excel_file>")
            sys.exit(1)

        if not os.path.exists(file_path):
            print(f"ERROR: File not found - {file_path}")
            sys.exit(1)
        
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"Loaded shape: {df.shape}")
        print(f"Dtypes:\n{df.dtypes}")
        print(f"Null counts:\n{df.isnull().sum()}")
        print(f"First 3 rows:\n{df.head(3)}")
        
        Path('artifacts').mkdir(exist_ok=True)
        output_path = 'artifacts/raw_data.pkl'
        joblib.dump(df, output_path)
        
        file_size = os.path.getsize(output_path)
        print(f"Saved to {output_path} ({file_size} bytes)")
        
    except Exception as e:
        print(f"ERROR in ingest: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    ingest_data(sys.argv[1] if len(sys.argv) > 1 else None)
