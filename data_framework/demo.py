import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch.autograd import Variable
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torchvision
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SVMSMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Generate synthetic data using sklearn's make_classification
X, y = make_classification(n_samples=20000,
                          n_features=40,
                          n_informative=20,
                          n_redundant=20,
                          weights=[0.98, 0.02],
                          random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Convert training data to a DataFrame
process = pd.DataFrame(X_train, columns=[f'fea{i}' for i in range(1, X_train.shape[1] + 1)])
process['target'] = y_train

# Extract relevant data for generating fake samples
X_for_generate = process.query("target == 1").iloc[:, :-1].values
X_non_default = process.query('target == 0').iloc[:, :-1].values
X_for_generate = torch.tensor(X_for_generate).type(torch.FloatTensor)

n_generate = X_non_default.shape[0] - X_for_generate.shape[0]

# Hyperparameters
BATCH_SIZE = 50
LR_G = 0.0001
LR_D = 0.0001
N_IDEAS = 20

# Build the generator G
G = nn.Sequential(
    nn.Linear(N_IDEAS, 128),
    nn.ReLU(),
    nn.Linear(128, 40),
)

# Build the discriminator D
D = nn.Sequential(
    nn.Linear(40, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)

# Define optimizers for D and G
opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

# GAN training
for step in range(100000):
    # Randomly select BATCH_SIZE real samples with a target of 1
    chosen_data = np.random.choice((X_for_generate.shape[0]), size=(BATCH_SIZE), replace=False)
    artist_paintings = X_for_generate[chosen_data, :]

    # Generate fake samples using the generator
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS, requires_grad=True)
    G_paintings = G(G_ideas)

    # Calculate the probabilities using the discriminator
    prob_artist1 = D(G_paintings)

    # Generator loss
    G_loss = torch.mean(torch.log(1. - prob_artist1))
    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    prob_artist0 = D(artist_paintings)
    prob_artist1 = D(G_paintings.detach())

    # Discriminator loss
    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()

# Generate fake data and combine with real data
fake_data = G(torch.randn(n_generate, N_IDEAS)).detach().numpy()
X_default = pd.DataFrame(np.concatenate([X_for_generate, fake_data]), columns=[f'fea{i}' for i in range(1, X_train.shape[1] + 1])
X_default['target'] = 1
X_non_default = pd.DataFrame(X_non_default, columns=[f'fea{i}' for i in range(1, X_train.shape[1] + 1])
X_non_default['target'] = 0
train_data_gan = pd.concat([X_default, X_non_default])

# Separate features and labels
X_gan = train_data_gan.iloc[:, :-1]
y_gan = train_data_gan.iloc[:, -1]

# Output the dimensions of the generated data
print(X_gan.shape, y_gan.shape)

# Train a Support Vector Machine (SVM) classifier on the generated data
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_gan, y_gan)

# Use SMOTE to oversample the training data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train a Random Forest classifier on the resampled data
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_resampled, y_resampled)

# Use EasyEnsemble to balance the class distribution
ee = EasyEnsembleClassifier(n_subsets=10, replacement=True, n_neighbors=3, n_jobs=-1)
ee.fit(X_train, y_train)

# Train a Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_classifier.fit(X_train, y_train)

# Use ADASYN to oversample the training data
adasyn = ADASYN(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Train an AdaBoost classifier on the resampled data
ab_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
ab_classifier.fit(X_resampled, y_resampled)

# Train a Logistic Regression classifier
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train, y_train)

# Train a Multi-layer Perceptron (MLP) classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train, y_train)

# Evaluate classifiers and print classification reports
svm_predictions = svm_classifier.predict(X_test)
print("Support Vector Machine Classifier Report:")
print(classification_report(y_test, svm_predictions))

rf_predictions = rf_classifier.predict(X_test)
print("Random Forest Classifier Report:")
print(classification_report(y_test, rf_predictions))

ee_predictions = ee.predict(X_test)
print("EasyEnsemble Classifier Report:")
print(classification_report(y_test, ee_predictions))

gb_predictions = gb_classifier.predict(X_test)
print("Gradient Boosting Classifier Report:")
print(classification_report(y_test, gb_predictions))

ab_predictions = ab_classifier.predict(X_test)
print("AdaBoost Classifier Report:")
print(classification_report(y_test, ab_predictions))

lr_predictions = lr_classifier.predict(X_test)
print("Logistic Regression Classifier Report:")
print(classification_report(y_test, lr_predictions))

mlp_predictions = mlp_classifier.predict(X_test)
print("Multi-layer Perceptron Classifier Report:")
print(classification_report(y_test, mlp_predictions))

# Plot a confusion matrix for one of the classifiers
conf_matrix = confusion_matrix(y_test, svm_predictions)
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title("Confusion Matrix for SVM Classifier")
plt.colorbar()

plt.xticks([0, 1], ["Predicted 0", "Predicted 1"])
plt.yticks([0, 1], ["Actual 0", "Actual 1"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i][j]), ha='center', va='center')

plt.show()
