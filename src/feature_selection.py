import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
import os

# Initialize variables to collect statistics across chunks
feature_importance_sum = {}
total_samples = 0

# Define the number of chunks you have
num_chunks = 506

for chunk_id in range(num_chunks):
    # Load the current chunk
    chunk_path = f'path_to_chunks/chunk_{chunk_id}.csv'  # Adjust path to your chunks
    
    if not os.path.exists(chunk_path):
        print(f"Skipping missing chunk {chunk_id}")
        continue
        
    print(f"Processing chunk {chunk_id}/{num_chunks}...")
    
    try:
        # Load chunk
        chunk_df = pd.read_csv(chunk_path)
        
        # Drop non-feature columns like IPs, timestamps, etc.
        columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 
                          'Destination IP', 'Destination Port', 'Timestamp', 'SimilarHTTP']
        
        # Only drop columns that exist
        columns_to_drop = [col for col in columns_to_drop if col in chunk_df.columns]
        chunk_df = chunk_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Convert to binary classification
        label_column = 'Label'  # Adjust to your label column name
        
        # Separate features and label
        X = chunk_df.drop(columns=[label_column])
        y = chunk_df[label_column]
        
        # Skip chunks with only one class
        if len(y.unique()) < 2:
            print(f"Skipping chunk {chunk_id} - contains only one class")
            continue
        
        # Handle missing values if any
        X = X.fillna(0)
        
        # Build a random forest model for this chunk
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Extract feature importances
        chunk_importances = model.feature_importances_
        
        # Add to running total
        for i, feature in enumerate(X.columns):
            if feature in feature_importance_sum:
                feature_importance_sum[feature] += chunk_importances[i] * len(X)
            else:
                feature_importance_sum[feature] = chunk_importances[i] * len(X)
        
        # Update total sample count
        total_samples += len(X)
        
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {str(e)}")

# Calculate average importance for each feature
average_importance = {feature: imp/total_samples for feature, imp in feature_importance_sum.items()}

# Sort features by importance
sorted_features = sorted(average_importance.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names = [feature for feature, _ in sorted_features]

print(f"\nTop 20 features by importance:")
for i, (feature, importance) in enumerate(sorted_features[:20]):
    print(f"{i+1}. {feature}: {importance:.6f}")

# Step 3: Evaluate performance with different feature counts
feature_counts = [5, 10, 15, 20, 30, 40, 50]  # Feature count options to try
cv_scores = []

# Choose a few chunks for cross-validation to save compute time
test_chunks = np.random.choice(num_chunks, size=min(5, num_chunks), replace=False)

for count in feature_counts:
    print(f"\nEvaluating with top {count} features...")
    selected_features = sorted_feature_names[:count]
    
    # Collect scores across test chunks
    chunk_scores = []
    
    for chunk_id in test_chunks:
        chunk_path = f'path_to_chunks/chunk_{chunk_id}.csv'
        
        if not os.path.exists(chunk_path):
            continue
            
        # Load chunk
        chunk_df = pd.read_csv(chunk_path)
        
        # Drop non-feature columns
        columns_to_drop = ['Flow ID', 'Source IP', 'Source Port', 
                          'Destination IP', 'Destination Port', 'Timestamp', 'SimilarHTTP']
        columns_to_drop = [col for col in columns_to_drop if col in chunk_df.columns]
        chunk_df = chunk_df.drop(columns=columns_to_drop, errors='ignore')
        
        # Prepare data
        X = chunk_df[selected_features].copy()  # Select only top features
        y = chunk_df[label_column]
        
        # Skip if chunk doesn't contain all needed features or has only one class
        if not all(feature in X.columns for feature in selected_features) or len(y.unique()) < 2:
            continue
            
        # Handle missing values
        X = X.fillna(0)
        
        # Perform cross-validation on this chunk
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X, y, cv=3, scoring='f1')
        chunk_scores.append(scores.mean())
        print(f"  Chunk {chunk_id} F1 score: {scores.mean():.4f}")
    
    # Average scores across chunks
    if chunk_scores:
        avg_score = np.mean(chunk_scores)
        cv_scores.append(avg_score)
        print(f"Average F1 score with {count} features: {avg_score:.4f}")
    else:
        print(f"No valid scores for {count} features")
        cv_scores.append(0)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(feature_counts, cv_scores, marker='o')
plt.xlabel('Number of Features')
plt.ylabel('Average F1 Score')
plt.title('Model Performance vs Number of Features')
plt.grid(True)
plt.savefig('feature_count_performance.png')
plt.show()

# Find optimal feature count
best_k = feature_counts[np.argmax(cv_scores)]
print(f"\nOptimal number of features: {best_k}")
print(f"Selected features: {sorted_feature_names[:best_k]}")

# Save the selected features to a file
with open('selected_features.txt', 'w') as f:
    for feature in sorted_feature_names[:best_k]:
        f.write(f"{feature}\n")

print(f"\nSelected features saved to 'selected_features.txt'")
