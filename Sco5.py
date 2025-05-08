print("📌 Missing Values in Each Column:\n")
print(df.isnull().sum())

# 📌 Total number of missing values
print("\n🧮 Total Missing Values:", df.isnull().sum().sum())

# 🔍 Check for duplicate rows
duplicate_rows = df.duplicated()
num_duplicates = duplicate_rows.sum()

print("\n📌 Total Duplicate Rows:", num_duplicates)

# Optional: View duplicate rows (uncomment if needed)
# print(df[df.duplicated()])
