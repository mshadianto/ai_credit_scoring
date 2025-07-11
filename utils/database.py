"""
Database utilities for AI Credit Scoring System
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import settings, get_db_url

Base = declarative_base()

class CreditApplication(Base):
    """Credit application model"""
    __tablename__ = 'credit_applications'
    
    id = Column(Integer, primary_key=True)
    application_date = Column(DateTime, default=datetime.utcnow)
    applicant_name = Column(String(255))
    age = Column(Integer)
    income = Column(Float)
    employment_years = Column(Integer)
    debt_to_income = Column(Float)
    credit_history_length = Column(Integer)
    number_of_accounts = Column(Integer)
    payment_history = Column(Float)
    credit_utilization = Column(Float)
    education_level = Column(String(100))
    marital_status = Column(String(50))
    home_ownership = Column(String(50))
    credit_type = Column(String(100))
    loan_amount = Column(Float)
    loan_purpose = Column(String(200))
    credit_score = Column(Integer)
    risk_probability = Column(Float)
    risk_level = Column(String(50))
    decision = Column(String(50))
    interest_rate = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ModelPerformance(Base):
    """Model performance tracking"""
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True)
    model_name = Column(String(100))
    credit_type = Column(String(100))
    accuracy = Column(Float)
    auc_score = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_date = Column(DateTime, default=datetime.utcnow)
    model_params = Column(Text)  # JSON string
    feature_importance = Column(Text)  # JSON string

class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.engine = create_engine(get_db_url(), echo=settings.DATABASE_ECHO)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session"""
        return self.SessionLocal()
    
    def save_credit_application(self, application_data: Dict) -> int:
        """Save credit application to database"""
        session = self.get_session()
        try:
            application = CreditApplication(**application_data)
            session.add(application)
            session.commit()
            application_id = application.id
            return application_id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_credit_applications(self, limit: int = 100, credit_type: Optional[str] = None) -> pd.DataFrame:
        """Get credit applications as DataFrame"""
        session = self.get_session()
        try:
            query = session.query(CreditApplication)
            if credit_type:
                query = query.filter(CreditApplication.credit_type == credit_type)
            
            applications = query.order_by(CreditApplication.created_at.desc()).limit(limit).all()
            
            # Convert to DataFrame
            data = []
            for app in applications:
                data.append({
                    'id': app.id,
                    'application_date': app.application_date,
                    'age': app.age,
                    'income': app.income,
                    'employment_years': app.employment_years,
                    'debt_to_income': app.debt_to_income,
                    'credit_score': app.credit_score,
                    'risk_level': app.risk_level,
                    'decision': app.decision,
                    'credit_type': app.credit_type,
                    'loan_amount': app.loan_amount
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def save_model_performance(self, performance_data: Dict) -> int:
        """Save model performance metrics"""
        session = self.get_session()
        try:
            # Convert dict fields to JSON strings
            if 'model_params' in performance_data:
                performance_data['model_params'] = json.dumps(performance_data['model_params'])
            if 'feature_importance' in performance_data:
                performance_data['feature_importance'] = json.dumps(performance_data['feature_importance'])
            
            performance = ModelPerformance(**performance_data)
            session.add(performance)
            session.commit()
            return performance.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_model_performance_history(self, model_name: str, credit_type: str) -> pd.DataFrame:
        """Get model performance history"""
        session = self.get_session()
        try:
            performances = session.query(ModelPerformance).filter(
                ModelPerformance.model_name == model_name,
                ModelPerformance.credit_type == credit_type
            ).order_by(ModelPerformance.training_date.desc()).all()
            
            data = []
            for perf in performances:
                data.append({
                    'training_date': perf.training_date,
                    'accuracy': perf.accuracy,
                    'auc_score': perf.auc_score,
                    'precision': perf.precision,
                    'recall': perf.recall,
                    'f1_score': perf.f1_score
                })
            
            return pd.DataFrame(data)
        finally:
            session.close()
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio-level metrics"""
        session = self.get_session()
        try:
            total_applications = session.query(CreditApplication).count()
            approved_applications = session.query(CreditApplication).filter(
                CreditApplication.decision == 'Approved'
            ).count()
            
            avg_credit_score = session.query(CreditApplication.credit_score).filter(
                CreditApplication.credit_score.isnot(None)
            ).all()
            
            if avg_credit_score:
                avg_score = sum([score[0] for score in avg_credit_score]) / len(avg_credit_score)
            else:
                avg_score = 0
            
            approval_rate = (approved_applications / total_applications * 100) if total_applications > 0 else 0
            
            return {
                'total_applications': total_applications,
                'approved_applications': approved_applications,
                'approval_rate': approval_rate,
                'average_credit_score': round(avg_score, 0)
            }
        finally:
            session.close()
    
    def export_data(self, table_name: str, output_path: str):
        """Export table data to CSV"""
        session = self.get_session()
        try:
            if table_name == 'credit_applications':
                df = pd.read_sql_query(session.query(CreditApplication).statement, self.engine)
            elif table_name == 'model_performance':
                df = pd.read_sql_query(session.query(ModelPerformance).statement, self.engine)
            else:
                raise ValueError(f"Unknown table: {table_name}")
            
            df.to_csv(output_path, index=False)
            return len(df)
        finally:
            session.close()

class DataGenerator:
    """Generate sample data for testing"""
    
    @staticmethod
    def generate_sample_applications(n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample credit applications"""
        import numpy as np
        
        np.random.seed(42)
        
        # Generate base features
        data = {
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.lognormal(10, 1, n_samples) * 1000000,  # IDR
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income': np.random.uniform(0, 0.8, n_samples),
            'credit_history_length': np.random.randint(0, 30, n_samples),
            'number_of_accounts': np.random.randint(1, 20, n_samples),
            'payment_history': np.random.uniform(0.5, 1, n_samples),
            'credit_utilization': np.random.uniform(0, 1, n_samples),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
            'home_ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples),
            'credit_type': np.random.choice(settings.CREDIT_TYPES, n_samples),
            'loan_amount': np.random.lognormal(11, 1, n_samples) * 1000000,  # IDR
            'loan_purpose': np.random.choice([
                'Home Purchase', 'Debt Consolidation', 'Business', 'Education', 'Other'
            ], n_samples)
        }
        
        # Generate synthetic risk scores
        risk_score = (
            -0.3 * (data['age'] - 25) / 55 +
            -0.4 * np.log(data['income']) / 15 +
            -0.2 * data['employment_years'] / 40 +
            0.6 * data['debt_to_income'] +
            -0.3 * data['credit_history_length'] / 30 +
            -0.1 * data['payment_history'] +
            0.4 * data['credit_utilization'] +
            np.random.normal(0, 0.3, n_samples)
        )
        
        # Convert to credit scores and decisions
        normalized_risk = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
        data['credit_score'] = (300 + (1 - normalized_risk) * 550).astype(int)
        data['risk_probability'] = normalized_risk
        
        # Risk levels
        risk_levels = []
        decisions = []
        interest_rates = []
        
        for score in data['credit_score']:
            if score >= 740:
                risk_levels.append('Very Low')
                decisions.append('Approved')
                interest_rates.append('5.5% - 7.0%')
            elif score >= 670:
                risk_levels.append('Low')
                decisions.append('Approved')
                interest_rates.append('7.0% - 9.5%')
            elif score >= 580:
                risk_levels.append('Medium')
                decisions.append('Review')
                interest_rates.append('9.5% - 15.0%')
            else:
                risk_levels.append('High')
                decisions.append('Rejected')
                interest_rates.append('N/A')
        
        data['risk_level'] = risk_levels
        data['decision'] = decisions
        data['interest_rate'] = interest_rates
        data['application_date'] = pd.date_range(
            start='2023-01-01', 
            end='2024-12-31', 
            periods=n_samples
        )
        
        return pd.DataFrame(data)

# Global database manager instance
db_manager = DatabaseManager()

def init_sample_data():
    """Initialize database with sample data"""
    try:
        # Check if data already exists
        existing_count = db_manager.get_session().query(CreditApplication).count()
        if existing_count > 0:
            print(f"Database already contains {existing_count} records")
            return
        
        # Generate and save sample data
        print("Generating sample data...")
        sample_data = DataGenerator.generate_sample_applications(1000)
        
        session = db_manager.get_session()
        for _, row in sample_data.iterrows():
            application = CreditApplication(
                application_date=row['application_date'],
                age=row['age'],
                income=row['income'],
                employment_years=row['employment_years'],
                debt_to_income=row['debt_to_income'],
                credit_history_length=row['credit_history_length'],
                number_of_accounts=row['number_of_accounts'],
                payment_history=row['payment_history'],
                credit_utilization=row['credit_utilization'],
                education_level=row['education_level'],
                marital_status=row['marital_status'],
                home_ownership=row['home_ownership'],
                credit_type=row['credit_type'],
                loan_amount=row['loan_amount'],
                loan_purpose=row['loan_purpose'],
                credit_score=row['credit_score'],
                risk_probability=row['risk_probability'],
                risk_level=row['risk_level'],
                decision=row['decision'],
                interest_rate=row['interest_rate']
            )
            session.add(application)
        
        session.commit()
        session.close()
        print("Sample data initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing sample data: {e}")

if __name__ == "__main__":
    init_sample_data()