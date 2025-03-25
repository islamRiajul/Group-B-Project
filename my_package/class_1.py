from sklearn.preprocessing import OneHotEncoder

# Dataset Preprocessing Class
class DataPreprocessor:
    def __init__(self, df):
        self.df = df
    
    def encode_categorical(self):
        # Encodes categorical columns using predefined mappings.
        building_t = {'Residential': 1, 'Commercial': 2, 'Industrial': 3}
        day_of_week = {'Weekend': 0, 'Weekday': 1}
        
        if "Building Type" in self.df.columns:
            self.df["Building Type"] = self.df["Building Type"].map(building_t)

        if "Day of Week" in self.df.columns:
            self.df["Day of Week"] = self.df["Day of Week"].map(day_of_week)
        
        return self.df
    
    def get_features_target(self, target_column="Energy Consumption"):
        # Splits dataset into features and target variable.
        if target_column not in self.df.columns:
            print(f"Error: The target column '{target_column}' is not found in the dataset.")
            return None, None
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return X, y