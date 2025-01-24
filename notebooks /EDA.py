import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
