# Required Libraries
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess
from xgboost import XGBRegressor
import pandas as pd
import matplotlib.pyplot as plt

# Define the BoostedHybrid class
class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.y_columns = None

    def fit(self, X_1, X_2, y):
        # Fit the first model
        self.model_1.fit(X_1, y)

        # Predict and compute residuals
        y_fit = pd.DataFrame(
            self.model_1.predict(X_1), index=X_1.index, columns=y.columns
        )
        y_resid = (y - y_fit).stack()

        # Fit the second model on residuals
        self.model_2.fit(X_2, y_resid)

        # Save column names for predictions
        self.y_columns = y.columns
        self.y_fit = y_fit
        self.y_resid = y_resid

    def predict(self, X_1, X_2):
        # Predict using the first model
        y_pred = pd.DataFrame(
            self.model_1.predict(X_1), index=X_1.index, columns=self.y_columns
        )
        y_pred = y_pred.stack()

        # Predict using the second model and align index
        y_resid_pred = pd.Series(self.model_2.predict(X_2), index=X_2.index)
        y_pred += y_resid_pred

        return y_pred.unstack()

# Load the data
comp_dir = Path('../input/store-sales-time-series-forecasting')
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales['date'] = store_sales.date.dt.to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family', 'date']).sort_index()

# Prepare the target series
y = store_sales.unstack(['store_nbr', 'family'])['sales'].loc['2017']

# Create training data
fourier = CalendarFourier(freq='M', order=4)
dp = DeterministicProcess(
    index=y.index,
    constant=True,
    order=1,
    seasonal=True,
    additional_terms=[fourier],
    drop=True,
)
X_1 = dp.in_sample()
X_1['NewYear'] = (X_1.index.dayofyear == 1)

# Create additional features for the second model
X_2 = store_sales.reset_index()
X_2 = X_2[X_2['date'].dt.year == 2017]
X_2['family'] = LabelEncoder().fit_transform(X_2['family'])
X_2 = X_2.set_index(['date', 'store_nbr', 'family']).sort_index()
X_2 = X_2[['onpromotion']]

# Train the BoostedHybrid model
model_1 = LinearRegression(fit_intercept=False)
model_2 = XGBRegressor(objective='reg:squarederror', n_estimators=100)
hybrid_model = BoostedHybrid(model_1, model_2)
hybrid_model.fit(X_1, X_2, y)

# Make predictions on the training set
y_pred = hybrid_model.predict(X_1, X_2)

# Plot results for a specific store and family
STORE_NBR = '1'
FAMILY = 'PRODUCE'
ax = y.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(
    figsize=(10, 6), alpha=0.75, title=f'{FAMILY} Sales at Store {STORE_NBR}'
)
y_pred.loc(axis=1)['sales', STORE_NBR, FAMILY].plot(ax=ax, linestyle='--')
ax.legend(['Actual', 'Fitted'])
plt.show()

# Prepare the test data for submission
df_test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)
df_test['date'] = df_test.date.dt.to_period('D')
df_test = df_test.set_index(['store_nbr', 'family', 'date']).sort_index()

# Create test features
X_test_1 = dp.out_of_sample(steps=16)
X_test_1.index.name = 'date'
X_test_1['NewYear'] = (X_test_1.index.dayofyear == 1)
X_test_2 = df_test.reset_index()
X_test_2['family'] = LabelEncoder().fit_transform(X_test_2['family'])
X_test_2 = X_test_2.set_index(['date', 'store_nbr', 'family']).sort_index()
X_test_2 = X_test_2[['onpromotion']]

# Make predictions for the test set
y_submit = hybrid_model.predict(X_test_1, X_test_2)
y_submit = y_submit.stack(['store_nbr', 'family'])
y_submit = y_submit.join(df_test.id).reindex(columns=['id', 'sales'])

# Save the predictions to submission.csv
y_submit.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")