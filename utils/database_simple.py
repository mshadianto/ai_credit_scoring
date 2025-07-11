"""
Simple Database utilities for AI Credit Scoring System
Minimal dependencies version
"""
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Any

# Get project root
BASE_DIR = Path(__file__).parent.parent

class SimpleDatabase:
    """Simple SQLite database manager"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = BASE_DIR / "data" / "credit_scoring.db"
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.db_path = str(db_path)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create credit applications table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS credit_applications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    application_date TEXT,
                    age INTEGER,
                    income REAL,
                    employment_years INTEGER,
                    debt_to_income REAL,
                    credit_history_length INTEGER,
                    number_of_accounts INTEGER,
                    payment_history REAL,
                    credit_utilization REAL,
                    education_level TEXT,
                    marital_status TEXT,
                    home_ownership TEXT,
                    credit_type TEXT,
                    loan_amount REAL,
                    loan_purpose TEXT,
                    credit_score INTEGER,
                    risk_probability REAL,
                    risk_level TEXT,
                    decision TEXT,
                    interest_rate TEXT,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
    
    def save_application(self, data: Dict) -> int:
        """Save credit application"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Prepare data
            now = datetime.now().isoformat()
            data['created_at'] = now
            if 'application_date' not in data:
                data['application_date'] = now
            
            # Insert data
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            
            cursor.execute(f'''
                INSERT INTO credit_applications ({columns})
                VALUES ({placeholders})
            ''', list(data.values()))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_applications(self, limit: int = 100) -> pd.DataFrame:
        """Get credit applications"""
        with sqlite3.connect(self.db_path) as conn:
            query = f'''
                SELECT * FROM credit_applications 
                ORDER BY created_at DESC 
                LIMIT {limit}
            '''
            return pd.read_sql_query(query, conn)
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio metrics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total applications
            cursor.execute("SELECT COUNT(*) FROM credit_applications")
            total = cursor.fetchone()[0]
            
            # Approved applications
            cursor.execute("SELECT COUNT(*) FROM credit_applications WHERE decision = 'Approved'")
            approved = cursor.fetchone()[0]
            
            # Average credit score
            cursor.execute("SELECT AVG(credit_score) FROM credit_applications WHERE credit_score IS NOT NULL")
            avg_score = cursor.fetchone()[0] or 0
            
            approval_rate = (approved / total * 100) if total > 0 else 0
            
            return {
                'total_applications': total,
                'approved_applications': approved,
                'approval_rate': approval_rate,
                'average_credit_score': avg_score
            }

class DataGenerator:
    """Generate sample data for testing"""
    
    @staticmethod
    def generate_sample_applications(n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample credit applications"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(11, 0.8, n_samples),  # Indonesian Rupiah scale
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income': np.random.uniform(0, 0.8, n_samples),
            'credit_history_length': np.random.randint(0, 30, n_samples),
            'number_of_accounts': np.random.randint(1, 20, n_samples),
            'payment_history': np.random.uniform(0.5, 1, n_samples),
            'credit_utilization': np.random.uniform(0, 1, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
            'credit_type': np.random.choice([
                'Personal Loan', 'Mortgage', 'Credit Card', 'Auto Loan', 'Business Loan'
            ], n_samples),
            'loan_amount': np.random.lognormal(11.5, 0.8, n_samples),
            'loan_purpose': np.random.choice([
                'Home Purchase', 'Debt Consolidation', 'Business', 'Education', 'Other'
            ], n_samples)
        }
        
        # Generate synthetic scores
        risk_factors = (
            -0.02 * (data['age'] - 35) +
            -0.0000002 * data['income'] +
            -0.02 * data['employment_years'] +
            0.6 * data['debt_to_income'] +
            -0.02 * data['credit_history_length'] +
            -0.1 * data['payment_history'] +
            0.4 * data['credit_utilization'] +
            np.random.normal(0, 0.3, n_samples)
        )
        
        # Normalize and convert to credit scores
        risk_normalized = (risk_factors - risk_factors.min()) / (risk_factors.max() - risk_factors.min())
        data['credit_score'] = (300 + (1 - risk_normalized) * 550).astype(int)
        data['risk_probability'] = risk_normalized
        
        # Assign risk levels and decisions
        risk_levels = []
        decisions = []
        interest_rates = []
        
        for score in data['credit_score']:
            if score >= 740:
                risk_levels.append('Sangat Rendah')
                decisions.append('Approved')
                interest_rates.append('5.5% - 7.0%')
            elif score >= 670:
                risk_levels.append('Rendah')
                decisions.append('Approved')
                interest_rates.append('7.0% - 9.5%')
            elif score >= 580:
                risk_levels.append('Sedang')
                decisions.append('Review')
                interest_rates.append('9.5% - 15.0%')
            else:
                risk_levels.append('Tinggi')
                decisions.append('Rejected')
                interest_rates.append('N/A')
        
        data['risk_level'] = risk_levels
        data['decision'] = decisions
        data['interest_rate'] = interest_rates
        
        # Add timestamps
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', periods=n_samples)
        data['application_date'] = dates.strftime('%Y-%m-%d %H:%M:%S')
        
        return pd.DataFrame(data)

# Global instance
db_manager = SimpleDatabase()

def init_sample_data():
    """Initialize database with sample data"""
    try:
        # Check if data exists
        existing = db_manager.get_applications(limit=1)
        if len(existing) > 0:
            print(f"Database already contains data")
            return
        
        print("Generating sample data...")
        sample_data = DataGenerator.generate_sample_applications(500)
        
        # Save sample data
        for _, row in sample_data.iterrows():
            try:
                db_manager.save_application(row.to_dict())
            except Exception as e:
                print(f"Error saving row: {e}")
                continue
        
        print("✅ Sample data initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing sample data: {e}")

if __name__ == "__main__":
    init_sample_data()