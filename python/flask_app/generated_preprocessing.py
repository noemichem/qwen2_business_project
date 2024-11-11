import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from scipy.stats import iqr

def preprocess(df):
    # Handle Missing Values
    df['Id'].fillna(df['Id'].mean(), inplace=True)
    df['MSSubClass'].fillna(df['MSSubClass'].mean(), inplace=True)
    df['MSZoning'].fillna(df['MSZoning'].mode()[0], inplace=True)
    df['LotArea'].fillna(df['LotArea'].mean(), inplace=True)
    df['TotalBsmtSF'].fillna(df['TotalBsmtSF'].mean(), inplace=True)
    df['GrLivArea'].fillna(df['GrLivArea'].mean(), inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
    df['MasVnrArea'].fillna(df['MasVnrArea'].mean(), inplace=True)
    df['BsmtFullBath'].fillna(df['BsmtFullBath'].mean(), inplace=True)
    df['BsmtHalfBath'].fillna(df['BsmtHalfBath'].mean(), inplace=True)
    df['GarageCars'].fillna(df['GarageCars'].mean(), inplace=True)
    df['GarageArea'].fillna(df['GarageArea'].mean(), inplace=True)
    df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean(), inplace=True)
    df['PoolArea'].fillna(0, inplace=True)
    df['MiscVal'].fillna(0, inplace=True)
    df['MoSold'].fillna(df['MoSold'].mean(), inplace=True)
    df['YrSold'].fillna(df['YrSold'].mean(), inplace=True)

    # Encode Categorical Variables
    categorical_columns = [
        'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
        'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
        'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 
        'Electrical', 'KitchenQual', 'Functional', 'Fireplaces', 
        'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 
        'GarageCond', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 
        'SaleCondition'
    ]

    # Initialize OneHotEncoder and LabelEncoder
    one_hot_encoder = OneHotEncoder(drop='first', sparse=False)
    label_encoder = LabelEncoder()

    # Apply OneHotEncoder to categorical columns with more than two unique values
    for col in categorical_columns:
        if df[col].dtype == 'object' and df[col].nunique() > 2:
            # Apply one-hot encoding
            encoded = one_hot_encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(encoded, columns=one_hot_encoder.get_feature_names_out([col]))
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=[col], inplace=True)

        # Apply LabelEncoder to binary categorical variables (those with only 2 unique values)
        elif df[col].dtype == 'object' and df[col].nunique() == 2:
            df[col] = label_encoder.fit_transform(df[col])

    # Scale/Normalize numerical features using StandardScaler
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Handle Outliers using IQR method (Interquartile Range)
    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        iqr_value = iqr(df[col])

        lower_bound = Q1 - 1.5 * iqr_value
        upper_bound = Q3 + 1.5 * iqr_value

        # Clip the values to be within the IQR bounds
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df