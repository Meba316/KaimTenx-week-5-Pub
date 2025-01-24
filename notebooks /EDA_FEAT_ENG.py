import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('transactions.csv')

# Define the default proxy variable based on FraudResult
# Assuming fraud = 1 means potential default or risk
df['Default'] = df['FraudResult'].apply(lambda x: 1 if x == 1 else 0)

# Get general information about the dataset
df.info()

# View the first few rows
df.head()

# Summary statistics for numerical columns
df.describe()


# Plot histogram for numerical features
numerical_cols = ['Amount', 'Value']
df[numerical_cols].hist(bins=50, figsize=(10, 6))
plt.show()

# KDE plot for a single feature (e.g., Amount)
sns.kdeplot(df['Amount'], shade=True)
plt.title('Distribution of Transaction Amount')
plt.show()

# Count plot for categorical features
sns.countplot(x='ProductCategory', data=df)
plt.title('Distribution of Product Category')
plt.xticks(rotation=45)
plt.show()

# Correlation matrix
correlation_matrix = df.corr()

# Heatmap to visualize correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Check for missing values
df.isnull().sum()

# Box plot to detect outliers
sns.boxplot(x=df['Amount'])
plt.title('Boxplot of Transaction Amount')
plt.show()

# Create aggregate features for each customer
df['Total_Transaction_Amount'] = df.groupby('AccountId')['Amount'].transform('sum')
df['Average_Transaction_Amount'] = df.groupby('AccountId')['Amount'].transform('mean')
df['Transaction_Count'] = df.groupby('AccountId')['TransactionId'].transform('count')
df['Transaction_Amount_Std'] = df.groupby('AccountId')['Amount'].transform('std')

# Convert TransactionStartTime to datetime
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

# Extract date-time features
df['Transaction_Hour'] = df['TransactionStartTime'].dt.hour
df['Transaction_Day'] = df['TransactionStartTime'].dt.day
df['Transaction_Month'] = df['TransactionStartTime'].dt.month
df['Transaction_Year'] = df['TransactionStartTime'].dt.year

# One-Hot Encoding for ProductCategory
df = pd.get_dummies(df, columns=['ProductCategory'], drop_first=True)

# Label Encoding for ChannelId
le = LabelEncoder()
df['ChannelId_encoded'] = le.fit_transform(df['ChannelId'])

# Impute missing values for numerical features with the median
df.fillna(df.median(), inplace=True)

# For categorical variables, use mode or create a new "Missing" category
df['ProductCategory'].fillna(df['ProductCategory'].mode()[0], inplace=True)

# Standardize the numerical columns
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Normalize the numerical columns
scaler = MinMaxScaler()
df['Amount_normalized'] = scaler.fit_transform(df[['Amount']])

