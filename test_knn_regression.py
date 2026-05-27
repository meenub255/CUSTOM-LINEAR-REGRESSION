"""
Unit tests for KNNRegression implementation.
"""

import numpy as np
import pytest
from custom_linear_regression import KNNRegression
from custom_linear_regression.exceptions import NotFittedError, DataQualityError


class TestKNNRegressionBasics:
    """Test basic KNN regression functionality."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(50, 3)
        self.y_train = self.X_train[:, 0] * 2 + self.X_train[:, 1] * 1.5 + np.random.randn(50) * 0.1
        
        self.X_test = np.random.randn(20, 3)
        self.y_test = self.X_test[:, 0] * 2 + self.X_test[:, 1] * 1.5 + np.random.randn(20) * 0.1
    
    def test_fit_and_predict(self):
        """Test fit and predict."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        assert y_pred.shape == (20,)
        assert not np.any(np.isnan(y_pred))
    
    def test_score(self):
        """Test R² score computation."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        r2 = model.score(self.X_test, self.y_test)
        assert isinstance(r2, float)
        assert r2 >= -np.inf  # R² can be negative
    
    def test_n_features_in(self):
        """Test n_features_in_ property."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        assert model.n_features_in_ == 3
    
    def test_not_fitted_error(self):
        """Test NotFittedError when predicting before fit."""
        model = KNNRegression(k=5)
        
        with pytest.raises(NotFittedError):
            model.predict(self.X_test)
    
    def test_feature_mismatch(self):
        """Test error on feature count mismatch."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        X_wrong = np.random.randn(10, 5)  # 5 features instead of 3
        with pytest.raises(ValueError):
            model.predict(X_wrong)


class TestDistanceMetrics:
    """Test different distance metrics."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X_train = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.y_train = np.array([0.0, 1.0, 2.0, 3.0])
        self.X_test = np.array([[1.5, 1.5]])
    
    def test_euclidean_metric(self):
        """Test Euclidean distance metric."""
        model = KNNRegression(k=2, metric='euclidean', weights='uniform', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Should average the 2 nearest points: [1,1]=1.0 and [2,2]=2.0
        assert np.isclose(y_pred[0], 1.5, atol=0.1)
    
    def test_manhattan_metric(self):
        """Test Manhattan distance metric."""
        model = KNNRegression(k=2, metric='manhattan', weights='uniform', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Should still find [1,1] and [2,2] as nearest
        assert np.isclose(y_pred[0], 1.5, atol=0.1)
    
    def test_minkowski_metric(self):
        """Test Minkowski distance metric."""
        model = KNNRegression(k=2, metric='minkowski', p=2, weights='uniform', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        assert np.isclose(y_pred[0], 1.5, atol=0.1)


class TestWeightingStrategies:
    """Test different weighting strategies."""
    
    def setup_method(self):
        """Setup test data."""
        self.X_train = np.array([[0, 0], [1, 0], [2, 0]])
        self.y_train = np.array([0.0, 1.0, 10.0])  # Very different value for far point
        self.X_test = np.array([[0.5, 0]])
    
    def test_uniform_weights(self):
        """Test uniform weighting (simple average)."""
        model = KNNRegression(k=2, metric='euclidean', weights='uniform', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Neighbors: [0,0]=0.0 (dist=0.5) and [1,0]=1.0 (dist=0.5)
        # Uniform: (0.0 + 1.0) / 2 = 0.5
        assert np.isclose(y_pred[0], 0.5, atol=0.1)
    
    def test_distance_weights(self):
        """Test distance-based weighting."""
        model = KNNRegression(k=2, metric='euclidean', weights='distance', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # Neighbors: [0,0]=0.0 (dist=0.5) and [1,0]=1.0 (dist=0.5)
        # Same distances, so same as uniform
        assert np.isclose(y_pred[0], 0.5, atol=0.1)


class TestScaling:
    """Test feature scaling."""
    
    def setup_method(self):
        """Setup test data with different scales."""
        self.X_train = np.array([
            [0.1, 100],
            [0.2, 200],
            [0.3, 300],
            [0.4, 400],
        ])
        self.y_train = np.array([1.0, 2.0, 3.0, 4.0])
        self.X_test = np.array([[0.25, 250]])
    
    def test_with_scaling(self):
        """Test with feature scaling."""
        model = KNNRegression(k=2, metric='euclidean', weights='distance', scale=True)
        model.fit(self.X_train, self.y_train)
        y_pred_scaled = model.predict(self.X_test)
        
        assert not np.isnan(y_pred_scaled[0])
    
    def test_without_scaling(self):
        """Test without feature scaling."""
        model = KNNRegression(k=2, metric='euclidean', weights='distance', scale=False)
        model.fit(self.X_train, self.y_train)
        y_pred_unscaled = model.predict(self.X_test)
        
        assert not np.isnan(y_pred_unscaled[0])
    
    def test_scaling_affects_results(self):
        """Verify that scaling can affect predictions (features at different scales)."""
        model_scaled = KNNRegression(k=2, metric='euclidean', weights='distance', scale=True)
        model_scaled.fit(self.X_train, self.y_train)
        y_pred_scaled = model_scaled.predict(self.X_test)
        
        model_unscaled = KNNRegression(k=2, metric='euclidean', weights='distance', scale=False)
        model_unscaled.fit(self.X_train, self.y_train)
        y_pred_unscaled = model_unscaled.predict(self.X_test)
        
        # Results should differ due to scale difference
        assert not np.isclose(y_pred_scaled[0], y_pred_unscaled[0])


class TestMissingValues:
    """Test missing value handling."""
    
    def test_mean_imputation(self):
        """Test mean imputation strategy."""
        X_train = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, np.nan],  # Missing value
            [4.0, 4.0],
        ])
        y_train = np.array([1.0, 2.0, 3.0, 4.0])
        
        model = KNNRegression(k=2, missing_strategy='mean', drop_missing=False)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train[:2])
        assert not np.any(np.isnan(y_pred))
    
    def test_drop_missing(self):
        """Test dropping rows with missing values."""
        X_train = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, np.nan],
            [4.0, 4.0],
        ])
        y_train = np.array([1.0, 2.0, 3.0, 4.0])
        
        model = KNNRegression(k=2, missing_strategy='mean', drop_missing=True)
        model.fit(X_train, y_train)
        
        # Should have only 3 training samples
        assert model.X_train_.shape[0] == 3


class TestOutlierHandling:
    """Test outlier detection and handling."""
    
    def test_zscore_outlier_removal(self):
        """Test Z-score based outlier removal."""
        X_train = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 100.0],  # Outlier
        ])
        y_train = np.array([1.0, 2.0, 3.0, 500.0])
        
        model = KNNRegression(k=2, outlier_strategy='zscore', outlier_threshold=2.0, outlier_action='remove')
        model.fit(X_train, y_train)
        
        # Should have removed the outlier row
        assert model.X_train_.shape[0] == 3
    
    def test_iqr_outlier_clipping(self):
        """Test IQR-based outlier clipping."""
        X_train = np.array([
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [100.0, 100.0],
        ])
        y_train = np.array([1.0, 2.0, 3.0, 500.0])
        
        model = KNNRegression(k=2, outlier_strategy='iqr', outlier_threshold=1.5, outlier_action='clip')
        model.fit(X_train, y_train)
        
        # Should have clipped the outlier values
        assert model.X_train_.shape[0] == 4


class TestFeatureImportance:
    """Test feature importance computation."""
    
    def setup_method(self):
        """Setup test data."""
        np.random.seed(42)
        self.X_train = np.random.randn(50, 3)
        # First feature is most important
        self.y_train = self.X_train[:, 0] * 5 + np.random.randn(50) * 0.1
        
        self.X_test = np.random.randn(20, 3)
        self.y_test = self.X_test[:, 0] * 5 + np.random.randn(20) * 0.1
    
    def test_compute_feature_importances(self):
        """Test feature importance computation."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        importances = model.compute_feature_importances(self.X_test, self.y_test, n_repeats=3)
        
        assert importances.shape == (3,)
        assert np.all(importances >= 0)  # Importances should be non-negative
        
        # Feature 0 should be most important
        assert np.argmax(importances) == 0
    
    def test_feature_importances_stored(self):
        """Test that feature importances are stored in the model."""
        model = KNNRegression(k=5)
        model.fit(self.X_train, self.y_train)
        
        importances = model.compute_feature_importances(self.X_test, self.y_test)
        
        # Should be stored as an attribute
        assert hasattr(model, 'feature_importances_')
        assert np.allclose(model.feature_importances_, importances)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_k_equals_n_samples(self):
        """Test when k equals number of training samples."""
        X_train = np.array([[0, 0], [1, 1], [2, 2]])
        y_train = np.array([0.0, 1.0, 2.0])
        X_test = np.array([[0.5, 0.5]])
        
        model = KNNRegression(k=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Should average all 3 neighbors
        assert np.isclose(y_pred[0], 1.0)
    
    def test_k_greater_than_n_samples(self):
        """Test when k is greater than number of training samples."""
        X_train = np.array([[0, 0], [1, 1]])
        y_train = np.array([0.0, 1.0])
        X_test = np.array([[0.5, 0.5]])
        
        model = KNNRegression(k=10)  # k > 2
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Should use only 2 neighbors
        assert not np.isnan(y_pred[0])
    
    def test_single_feature(self):
        """Test with single feature."""
        X_train = np.array([[0.0], [1.0], [2.0]])
        y_train = np.array([0.0, 1.0, 2.0])
        X_test = np.array([[0.5]])
        
        model = KNNRegression(k=2)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        assert np.isclose(y_pred[0], 0.5)


class TestParameterValidation:
    """Test parameter validation."""
    
    def test_invalid_k(self):
        """Test invalid k value."""
        with pytest.raises(ValueError):
            KNNRegression(k=0)
        
        with pytest.raises(ValueError):
            KNNRegression(k=-1)
    
    def test_invalid_metric(self):
        """Test invalid metric."""
        with pytest.raises(ValueError):
            KNNRegression(metric='invalid_metric')
    
    def test_invalid_weights(self):
        """Test invalid weights."""
        with pytest.raises(ValueError):
            KNNRegression(weights='invalid_weights')
    
    def test_invalid_missing_strategy(self):
        """Test invalid missing strategy."""
        with pytest.raises(ValueError):
            KNNRegression(missing_strategy='invalid_strategy')
    
    def test_invalid_outlier_strategy(self):
        """Test invalid outlier strategy."""
        with pytest.raises(ValueError):
            KNNRegression(outlier_strategy='invalid_strategy')


class TestRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__ method."""
        model = KNNRegression(k=5, metric='euclidean', weights='distance')
        repr_str = repr(model)
        
        assert 'KNNRegression' in repr_str
        assert 'k=5' in repr_str
        assert 'euclidean' in repr_str
        assert 'distance' in repr_str


class TestGetParams:
    """Test get_params method."""
    
    def test_get_params(self):
        """Test get_params method."""
        model = KNNRegression(
            k=7,
            metric='manhattan',
            weights='distance',
            p=2,
            scale=False,
            missing_strategy='median',
            drop_missing=True,
            outlier_strategy='iqr',
            outlier_threshold=2.0,
            outlier_action='clip'
        )
        
        params = model.get_params()
        
        assert params['k'] == 7
        assert params['metric'] == 'manhattan'
        assert params['weights'] == 'distance'
        assert params['p'] == 2
        assert params['scale'] is False
        assert params['missing_strategy'] == 'median'
        assert params['drop_missing'] is True
        assert params['outlier_strategy'] == 'iqr'
        assert params['outlier_threshold'] == 2.0
        assert params['outlier_action'] == 'clip'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
