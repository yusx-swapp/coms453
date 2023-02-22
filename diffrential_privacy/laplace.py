import pandas as pd
import numpy as np
from scipy import stats



# Load the Adult dataset from the UCI Machine Learning Repository
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                 header=None, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                     'marital-status', 'occupation', 'relationship', 'race', 'sex',
                                     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                                     'income'])

# Query the average age of the records with age greater than 25
result = df[df['age'] > 25]['age'].mean()
print('Original query result:', result)

# Set the privacy parameter epsilon
epsilon = 1.0

# Set the sensitivity of the query
sensitivity = 1.0

# Calculate the scale parameter for the Laplace distribution
scale = sensitivity / epsilon

# Generate the Laplacian noise
noise = np.random.laplace(loc=0, scale=scale)

# Add the noise to the query result to obtain the private result
private_result = result + noise

print('Private query result (epsilon=1):', private_result)

# Set a lower value of epsilon for 1-differential privacy
epsilon = 0.5

# Calculate the new scale parameter for the Laplace distribution
scale = sensitivity / epsilon

# Generate new Laplacian noise
noise = np.random.laplace(loc=0, scale=scale)

# Add the new noise to the original result to obtain the new private result
private_result = result + noise

print('Private query result (epsilon=0.5):', private_result)
