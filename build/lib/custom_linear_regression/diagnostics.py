"""
Diagnostic tools for custom linear regression models.

This module provides tests for normality, multicollinearity,
heteroscedasticity, and basic outlier detection.
"""

import warnings

import numpy as np
from scipy import stats


def _as_2d_float_array(X):
    X = np.array(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    return X


def _feature_labels(n_features, feature_names=None):
    if feature_names is not None and len(feature_names) == n_features:
        return list(feature_names)
    return [f"Feature_{index}" for index in range(n_features)]


class HeteroscedasticityTest:
    """
    Test for heteroscedasticity (non-constant residual variance).
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}

    def breusch_pagan_test(self, residuals, X):
        residuals = np.array(residuals, dtype=float)
        X = _as_2d_float_array(X)
        n_samples = len(residuals)

        residuals_squared = residuals ** 2
        X_aug = np.c_[np.ones(n_samples), X]

        beta = np.linalg.lstsq(X_aug, residuals_squared, rcond=None)[0]
        y_pred = np.dot(X_aug, beta)
        ss_res = np.sum((residuals_squared - y_pred) ** 2)
        ss_tot = np.sum((residuals_squared - np.mean(residuals_squared)) ** 2)
        r_squared = 0.0 if np.isclose(ss_tot, 0.0) else 1 - (ss_res / ss_tot)

        statistic = n_samples * max(r_squared, 0.0)
        p_value = 1 - stats.chi2.cdf(statistic, df=X.shape[1])
        is_homoscedastic = p_value > self.alpha

        result = {
            "test": "Breusch-Pagan",
            "statistic": statistic,
            "p_value": p_value,
            "is_homoscedastic": is_homoscedastic,
            "interpretation": "Homoscedastic" if is_homoscedastic else "Heteroscedastic",
        }
        self.results["breusch_pagan"] = result
        return result

    def white_test(self, residuals, X):
        residuals = np.array(residuals, dtype=float)
        X = _as_2d_float_array(X)
        n_samples, n_features = X.shape

        auxiliary_terms = [np.ones(n_samples)]
        for feature_index in range(n_features):
            auxiliary_terms.append(X[:, feature_index])
        for feature_index in range(n_features):
            auxiliary_terms.append(X[:, feature_index] ** 2)
        for left_index in range(n_features):
            for right_index in range(left_index + 1, n_features):
                auxiliary_terms.append(X[:, left_index] * X[:, right_index])

        X_aux = np.column_stack(auxiliary_terms)
        residuals_squared = residuals ** 2

        beta = np.linalg.lstsq(X_aux, residuals_squared, rcond=None)[0]
        y_pred = np.dot(X_aux, beta)
        ss_res = np.sum((residuals_squared - y_pred) ** 2)
        ss_tot = np.sum((residuals_squared - np.mean(residuals_squared)) ** 2)
        r_squared = 0.0 if np.isclose(ss_tot, 0.0) else 1 - (ss_res / ss_tot)

        statistic = n_samples * max(r_squared, 0.0)
        p_value = 1 - stats.chi2.cdf(statistic, df=X_aux.shape[1] - 1)
        is_homoscedastic = p_value > self.alpha

        result = {
            "test": "White",
            "statistic": statistic,
            "p_value": p_value,
            "is_homoscedastic": is_homoscedastic,
            "interpretation": "Homoscedastic" if is_homoscedastic else "Heteroscedastic",
        }
        self.results["white"] = result
        return result

    def goldfeld_quandt_test(self, residuals, X, split_fraction=0.2):
        residuals = np.array(residuals, dtype=float)
        X = _as_2d_float_array(X)
        n_samples = len(residuals)
        if n_samples < 9:
            raise ValueError("Goldfeld-Quandt test requires at least 9 samples.")

        split_fraction = float(np.clip(split_fraction, 0.0, 0.8))
        sort_idx = np.argsort(X[:, 0])
        residuals_sorted = residuals[sort_idx]

        drop_count = int(n_samples * split_fraction)
        remaining = n_samples - drop_count
        group_size = remaining // 2
        if group_size < 2:
            raise ValueError("Goldfeld-Quandt split left too few samples per group.")

        left_group = residuals_sorted[:group_size]
        right_group = residuals_sorted[-group_size:]

        var_left = np.var(left_group, ddof=1)
        var_right = np.var(right_group, ddof=1)
        small = max(min(var_left, var_right), 1e-12)
        large = max(var_left, var_right)

        statistic = large / small
        if var_left >= var_right:
            df1, df2 = len(left_group) - 1, len(right_group) - 1
        else:
            df1, df2 = len(right_group) - 1, len(left_group) - 1

        p_value = 1 - stats.f.cdf(statistic, df1, df2)
        is_homoscedastic = p_value > self.alpha

        result = {
            "test": "Goldfeld-Quandt",
            "statistic": statistic,
            "p_value": p_value,
            "is_homoscedastic": is_homoscedastic,
            "interpretation": "Homoscedastic" if is_homoscedastic else "Heteroscedastic",
        }
        self.results["goldfeld_quandt"] = result
        return result

    def run_all_tests(self, residuals, X):
        self.breusch_pagan_test(residuals, X)
        self.white_test(residuals, X)
        self.goldfeld_quandt_test(residuals, X)
        return self.results

    def print_summary(self):
        print("\n" + "=" * 70)
        print("HETEROSCEDASTICITY TEST RESULTS")
        print("=" * 70)
        for result in self.results.values():
            print(f"\n{result['test']}:")
            print(f"  Statistic: {result['statistic']:.4f}")
            print(f"  P-value: {result['p_value']:.6f}")
            print(f"  Result: {result['interpretation']}")


class NormalityTest:
    """
    Test whether residuals are approximately normal.
    """

    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.results = {}

    def shapiro_wilk_test(self, residuals):
        residuals = np.array(residuals, dtype=float)
        statistic, p_value = stats.shapiro(residuals)
        result = {
            "test": "Shapiro-Wilk",
            "statistic": statistic,
            "p_value": p_value,
            "is_normal": p_value > self.alpha,
            "interpretation": "Normal" if p_value > self.alpha else "Not Normal",
        }
        self.results["shapiro_wilk"] = result
        return result

    def anderson_darling_test(self, residuals):
        residuals = np.array(residuals, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            result_obj = stats.anderson(residuals, dist="norm")
        critical_value = result_obj.critical_values[2]
        is_normal = result_obj.statistic < critical_value
        result = {
            "test": "Anderson-Darling",
            "statistic": result_obj.statistic,
            "critical_value": critical_value,
            "is_normal": is_normal,
            "interpretation": "Normal" if is_normal else "Not Normal",
        }
        self.results["anderson_darling"] = result
        return result

    def jarque_bera_test(self, residuals):
        residuals = np.array(residuals, dtype=float)
        statistic, p_value = stats.jarque_bera(residuals)
        result = {
            "test": "Jarque-Bera",
            "statistic": statistic,
            "p_value": p_value,
            "skewness": stats.skew(residuals),
            "kurtosis": stats.kurtosis(residuals),
            "is_normal": p_value > self.alpha,
            "interpretation": "Normal" if p_value > self.alpha else "Not Normal",
        }
        self.results["jarque_bera"] = result
        return result

    def kolmogorov_smirnov_test(self, residuals):
        residuals = np.array(residuals, dtype=float)
        residual_std = np.std(residuals)
        if np.isclose(residual_std, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            standardized = (residuals - np.mean(residuals)) / residual_std
            statistic, p_value = stats.kstest(standardized, "norm")

        result = {
            "test": "Kolmogorov-Smirnov",
            "statistic": statistic,
            "p_value": p_value,
            "is_normal": p_value > self.alpha,
            "interpretation": "Normal" if p_value > self.alpha else "Not Normal",
        }
        self.results["kolmogorov_smirnov"] = result
        return result

    def run_all_tests(self, residuals):
        self.shapiro_wilk_test(residuals)
        self.anderson_darling_test(residuals)
        self.jarque_bera_test(residuals)
        self.kolmogorov_smirnov_test(residuals)
        return self.results

    def print_summary(self):
        print("\n" + "=" * 70)
        print("NORMALITY TEST RESULTS")
        print("=" * 70)
        for result in self.results.values():
            print(f"\n{result['test']}:")
            print(f"  Statistic: {result['statistic']:.4f}")
            if "p_value" in result:
                print(f"  P-value: {result['p_value']:.6f}")
            if "critical_value" in result:
                print(f"  Critical Value: {result['critical_value']:.4f}")
            if "skewness" in result:
                print(f"  Skewness: {result['skewness']:.4f}")
                print(f"  Kurtosis: {result['kurtosis']:.4f}")
            print(f"  Result: {result['interpretation']}")


class MulticollinearityTest:
    """
    Test for multicollinearity among features.
    """

    def __init__(self, vif_threshold=5.0):
        self.vif_threshold = vif_threshold
        self.correlation_matrix = None
        self.vif_scores = None
        self.feature_names = None

    def _standardize(self, X):
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds = np.where(stds == 0, 1.0, stds)
        return (X - means) / stds

    def calculate_vif(self, X, feature_names=None):
        X = _as_2d_float_array(X)
        X_std = self._standardize(X)
        n_features = X_std.shape[1]
        labels = _feature_labels(n_features, feature_names)

        self.feature_names = labels
        self.correlation_matrix = np.corrcoef(X_std.T)
        self.vif_scores = {}

        for index in range(n_features):
            X_others = np.delete(X_std, index, axis=1)
            y_feature = X_std[:, index]
            X_reg = np.c_[np.ones(X_others.shape[0]), X_others]

            beta = np.linalg.lstsq(X_reg, y_feature, rcond=None)[0]
            y_pred = np.dot(X_reg, beta)
            ss_res = np.sum((y_feature - y_pred) ** 2)
            ss_tot = np.sum((y_feature - np.mean(y_feature)) ** 2)
            r_squared = 1.0 if np.isclose(ss_tot, 0.0) else 1 - (ss_res / ss_tot)
            vif = np.inf if r_squared >= 1.0 else 1 / (1 - r_squared)
            self.vif_scores[labels[index]] = vif

        return self.vif_scores

    def get_correlation_matrix(self, X):
        X = _as_2d_float_array(X)
        X_std = self._standardize(X)
        self.correlation_matrix = np.corrcoef(X_std.T)
        return self.correlation_matrix

    def detect_high_correlation_pairs(self, X, threshold=0.9):
        X = _as_2d_float_array(X)
        if self.correlation_matrix is None:
            self.get_correlation_matrix(X)

        labels = _feature_labels(self.correlation_matrix.shape[0], self.feature_names)
        high_pairs = []
        for left_index in range(self.correlation_matrix.shape[0]):
            for right_index in range(left_index + 1, self.correlation_matrix.shape[0]):
                corr = abs(self.correlation_matrix[left_index, right_index])
                if corr > threshold:
                    high_pairs.append((labels[left_index], labels[right_index], corr))
        return high_pairs

    def print_vif_summary(self):
        print("\n" + "=" * 70)
        print("VARIANCE INFLATION FACTOR (VIF) RESULTS")
        print("=" * 70)
        print(f"VIF Threshold: {self.vif_threshold}")
        print(f"(VIF > {self.vif_threshold} indicates problematic multicollinearity)")
        print("-" * 70)

        if self.vif_scores:
            for feature, vif in self.vif_scores.items():
                status = "PROBLEMATIC" if vif > self.vif_threshold else "OK"
                if np.isinf(vif):
                    print(f"{feature:20s}: INF (Perfect multicollinearity) {status}")
                else:
                    print(f"{feature:20s}: {vif:8.4f} {status}")

    def print_correlation_summary(self, X, threshold=0.9):
        print("\n" + "=" * 70)
        print("HIGH CORRELATION PAIRS")
        print("=" * 70)
        print(f"Threshold: {threshold}")
        print("-" * 70)

        high_pairs = self.detect_high_correlation_pairs(X, threshold=threshold)
        if high_pairs:
            for feat_i, feat_j, corr in high_pairs:
                print(f"{feat_i} <-> {feat_j}: {corr:.4f}")
        else:
            print("No high correlation pairs detected.")

    def run_all_tests(self, X, feature_names=None, corr_threshold=0.9):
        self.calculate_vif(X, feature_names)
        self.get_correlation_matrix(X)
        return {
            "vif_scores": self.vif_scores,
            "correlation_matrix": self.correlation_matrix,
            "high_correlation_pairs": self.detect_high_correlation_pairs(X, corr_threshold),
        }


class OutlierDetector:
    """
    Detect outliers in feature matrices using z-score or IQR rules.
    """

    def __init__(self, strategy="zscore", threshold=3.0):
        if strategy not in ["zscore", "iqr"]:
            raise ValueError("Strategy must be 'zscore' or 'iqr'")
        self.strategy = strategy
        self.threshold = threshold

    def detect(self, X, feature_names=None):
        X = _as_2d_float_array(X)
        labels = _feature_labels(X.shape[1], feature_names)

        if self.strategy == "zscore":
            means = np.mean(X, axis=0)
            stds = np.std(X, axis=0)
            stds = np.where(stds == 0, 1.0, stds)
            zscores = np.abs((X - means) / stds)
            mask = zscores > self.threshold
        else:
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            mask = (X < lower) | (X > upper)

        row_outlier_mask = mask.any(axis=1)
        feature_counts = {labels[index]: int(mask[:, index].sum()) for index in range(X.shape[1])}

        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "row_outlier_mask": row_outlier_mask,
            "n_outlier_rows": int(row_outlier_mask.sum()),
            "feature_outlier_counts": feature_counts,
        }


HeteroscedacityTest = HeteroscedasticityTest