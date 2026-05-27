class CustomLinearRegressionError(Exception):
    """Base class for exceptions in this library."""
    pass

class NotFittedError(CustomLinearRegressionError):
    """Exception raised when predict or score is called before fit."""
    def __init__(self, message="This LinearRegression instance is not fitted yet."):
        self.message = message
        super().__init__(self.message)

class OptimizationFailedError(CustomLinearRegressionError):
    """Exception raised when gradient descent diverges or fails to converge."""
    def __init__(self, message="Optimization failed to converge."):
        self.message = message
        super().__init__(self.message)

class DataQualityError(CustomLinearRegressionError):
    """Exception raised when data quality is too poor to proceed (e.g., all rows dropped)."""
    def __init__(self, message="Data quality issue prevented operation."):
        self.message = message
        super().__init__(self.message)
