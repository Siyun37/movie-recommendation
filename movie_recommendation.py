# %%
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# %%
# Load dataset
full_filepath = "https://raw.githubusercontent.com/Siyun37/movie-recommendation/main/IMDB-Movie-Data.csv"
data = pd.read_csv(full_filepath)
data.fillna(0, inplace=True)  # Replace missing values with 0


# %%
# Drop unnecessary columns
data = data.drop(['Rank', 'Description'], axis=1)

# Handle missing values (drop rows with missing values for simplicity)
data = data.dropna()


# %%
# Separate features and target
X = data.drop(['Rating', 'Title'], axis=1)
y = data['Rating']

# Encode categorical variables
categorical_features = ['Genre', 'Director', 'Actors']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ], remainder='passthrough'
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
# Build the Gradient Boosting model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)


# %%
# Compare predicted ratings with actual ratings on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Combine actual and predicted ratings for comparison
comparison = pd.DataFrame({
    'ActualRating': y_test.values,
    'PredictedRating': y_pred
})
print(comparison.head())


# %%
# Define a recommendation function
def recommend_movies(genre, actor, top_n=5):
    # Filter dataset by genre and actor
    filtered_data = data[data['Genre'].str.contains(genre, case=False, na=False) &
                         data['Actors'].str.contains(actor, case=False, na=False)]
    
    if filtered_data.empty:
        return "No movies found for the specified genre and actor."

    # Predict ratings for the filtered data
    filtered_X = filtered_data.drop(['Rating'], axis=1)
    filtered_data['PredictedRating'] = model.predict(filtered_X)

    # Sort by predicted rating and return top N movies
    recommendations = filtered_data.sort_values(by='PredictedRating', ascending=False).head(top_n)
    return recommendations[['Title', 'Genre', 'Actors', 'PredictedRating']]

# Example usage
result = recommend_movies(genre='Drama', actor='Kate Winslet', top_n=5)
print(result)


# %%



