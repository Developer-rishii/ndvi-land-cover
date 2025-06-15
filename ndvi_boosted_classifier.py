import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.signal import savgol_filter

# Load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

ndvi_cols = [col for col in train.columns if '_N' in col]

# Interpolate missing values
train[ndvi_cols] = train[ndvi_cols].interpolate(axis=1, limit_direction='both')
test[ndvi_cols] = test[ndvi_cols].interpolate(axis=1, limit_direction='both')

# Denoise using Savitzky-Golay filter
def denoise(row):
    filtered = savgol_filter(row, 5, 2)
    return pd.Series(filtered, index=row.index)

train[ndvi_cols] = train[ndvi_cols].apply(denoise, axis=1)
test[ndvi_cols] = test[ndvi_cols].apply(denoise, axis=1)

# Feature Engineering
def extract_features(row):
    values = row.values
    first_half = values[:len(values)//2]
    second_half = values[len(values)//2:]
    diff = np.diff(values)
    trend = np.polyfit(range(len(values)), values, 1)[0]
    grad = np.gradient(values)
    
    return pd.Series({
        'mean': values.mean(),
        'std': values.std(),
        'max': values.max(),
        'min': values.min(),
        'range': values.max() - values.min(),
        'trend': trend,
        'mean_first': first_half.mean(),
        'mean_second': second_half.mean(),
        'std_diff': diff.std(),
        'max_diff': diff.max(),
        'min_diff': diff.min(),
        'grad_mean': grad.mean(),
        'grad_std': grad.std(),
        'grad_max': grad.max(),
        'grad_min': grad.min()
    })

# Apply features
X_train_feats = train[ndvi_cols].apply(extract_features, axis=1)
X_test_feats = test[ndvi_cols].apply(extract_features, axis=1)

# PCA for temporal signal
pca = PCA(n_components=5)
pca_train = pd.DataFrame(pca.fit_transform(train[ndvi_cols]), columns=[f'pca_{i}' for i in range(5)])
pca_test = pd.DataFrame(pca.transform(test[ndvi_cols]), columns=[f'pca_{i}' for i in range(5)])

# Combine all features
X_train = pd.concat([X_train_feats, pca_train], axis=1)
X_test = pd.concat([X_test_feats, pca_test], axis=1)

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train['class'])

# Train model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# Optional: cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# Predict
preds = model.predict(X_test)
labels = le.inverse_transform(preds)

# Output file
submission = pd.DataFrame({
    'ID': test['ID'],
    'class': labels
})
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created.")

