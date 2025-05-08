# Show all column names
print("🧾 Column Names:")
print(df.columns.tolist())

# Identify target column
target_col = 'label'
print("\n🎯 Target Variable:")
print(target_col)

# Identify feature columns (drop target)
feature_cols = df.drop(columns=[target_col]).columns.tolist()
print("\n🧰 Feature Columns:")
print(feature_cols)
