"""
Unit tests for ML utilities
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.ml_utils import FeatureEngineer, ModelEnsemble
from utils.database import DataGenerator

class TestFeatureEngineer:
    """Test feature engineering functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.feature_engineer = FeatureEngineer()
        self.sample_data = DataGenerator.generate_sample_applications(100)
    
    def test_create_features(self):
        """Test feature creation"""
        original_cols = len(self.sample_data.columns)
        enhanced_data = self.feature_engineer.create_features(self.sample_data)
        
        # Should have more features
        assert len(enhanced_data.columns) > original_cols
        
        # Check specific new features
        expected_features = [
            'income_to_age_ratio',
            'debt_per_account',
            'credit_density',
            'high_utilization',
            'young_borrower',
            'credit_behavior_score'
        ]
        
        for feature in expected_features:
            assert feature in enhanced_data.columns
    
    def test_prepare_features(self):
        """Test feature preparation for training"""
        # Add target variable
        self.sample_data['default_risk'] = np.random.binomial(1, 0.3, len(self.sample_data))
        
        X, y = self.feature_engineer.prepare_features(self.sample_data, 'default_risk')
        
        # Check shapes
        assert len(X) == len(self.sample_data)
        assert len(y) == len(self.sample_data)
        assert 'default_risk' not in X.columns
        
        # Check data types (should be numeric after encoding)
        assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
    
    def test_feature_scaling(self):
        """Test feature scaling"""
        X, _ = self.feature_engineer.prepare_features(self.sample_data)
        
        # Numerical features should be roughly standardized
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            mean_val = X[col].mean()
            std_val = X[col].std()
            assert abs(mean_val) < 1e-10  # Close to 0
            assert abs(std_val - 1) < 1e-10  # Close to 1

class TestModelEnsemble:
    """Test model ensemble functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.model_ensemble = ModelEnsemble()
        self.sample_data = DataGenerator.generate_sample_applications(200)
        self.sample_data['default_risk'] = np.random.binomial(1, 0.3, len(self.sample_data))
        self.credit_type = "Test_Loan"
    
    def test_train_models(self):
        """Test model training"""
        results = self.model_ensemble.train_models(
            self.sample_data, 
            self.credit_type, 
            target_col='default_risk'
        )
        
        # Should have results for multiple models
        assert len(results) > 0
        
        # Check metrics exist
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
                for metric in expected_metrics:
                    assert metric in metrics
                    assert 0 <= metrics[metric] <= 1
    
    def test_predict_ensemble(self):
        """Test ensemble prediction"""
        # Train models first
        self.model_ensemble.train_models(
            self.sample_data, 
            self.credit_type, 
            target_col='default_risk'
        )
        
        # Test prediction
        test_features = {
            'age': 35,
            'income': 500_000_000,
            'employment_years': 5,
            'debt_to_income': 0.3,
            'credit_history_length': 10,
            'number_of_accounts': 5,
            'payment_history': 0.85,
            'credit_utilization': 0.3,
            'education_level': 'Bachelor',
            'marital_status': 'Married',
            'home_ownership': 'Own'
        }
        
        prediction = self.model_ensemble.predict_ensemble(test_features, self.credit_type)
        
        # Check prediction structure
        assert 'credit_score' in prediction
        assert 'risk_probability' in prediction
        assert 'risk_level' in prediction
        assert 'decision' in prediction
        
        # Check value ranges
        assert 300 <= prediction['credit_score'] <= 850
        assert 0 <= prediction['risk_probability'] <= 1
        assert prediction['risk_level'] in ['Sangat Rendah', 'Rendah', 'Sedang', 'Tinggi', 'Sangat Tinggi']
        assert prediction['decision'] in ['Approved', 'Review', 'Rejected']
    
    def test_risk_level_mapping(self):
        """Test risk level mapping"""
        test_cases = [
            (0.1, 'Sangat Rendah'),
            (0.3, 'Rendah'),
            (0.5, 'Sedang'),
            (0.7, 'Tinggi'),
            (0.9, 'Sangat Tinggi')
        ]
        
        for risk_prob, expected_level in test_cases:
            level = self.model_ensemble._get_risk_level(risk_prob)
            assert level == expected_level

class TestDataGenerator:
    """Test data generation utilities"""
    
    def test_generate_sample_applications(self):
        """Test sample data generation"""
        n_samples = 500
        data = DataGenerator.generate_sample_applications(n_samples)
        
        # Check shape
        assert len(data) == n_samples
        assert len(data.columns) > 10
        
        # Check required columns
        required_cols = [
            'age', 'income', 'employment_years', 'debt_to_income',
            'credit_history_length', 'payment_history', 'credit_utilization',
            'education_level', 'marital_status', 'home_ownership',
            'credit_score', 'risk_level', 'decision'
        ]
        
        for col in required_cols:
            assert col in data.columns
        
        # Check value ranges
        assert data['age'].between(18, 80).all()
        assert data['debt_to_income'].between(0, 1).all()
        assert data['payment_history'].between(0, 1).all()
        assert data['credit_utilization'].between(0, 1).all()
        assert data['credit_score'].between(300, 850).all()
    
    def test_data_consistency(self):
        """Test data consistency"""
        data = DataGenerator.generate_sample_applications(100)
        
        # High credit scores should have low risk
        high_score_data = data[data['credit_score'] >= 740]
        if len(high_score_data) > 0:
            low_risk_ratio = (high_score_data['risk_level'] == 'Very Low').mean()
            assert low_risk_ratio > 0.5  # Most should be low risk
        
        # High debt-to-income should correlate with higher risk
        high_dti = data[data['debt_to_income'] > 0.6]
        if len(high_dti) > 0:
            high_risk_ratio = high_dti['risk_level'].isin(['High', 'Very High']).mean()
            # Should have some correlation, but not strict

if __name__ == "__main__":
    pytest.main([__file__])