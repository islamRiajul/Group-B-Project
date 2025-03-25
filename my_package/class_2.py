from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train(self, X_train, y_train):
        # Trains the linear model.
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        # Makes predictions on test data.
        return self.model.predict(X_test)
    
    def evaluate(self, y_true, y_pred):
        # Computes MSE and R^2 Score.
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2