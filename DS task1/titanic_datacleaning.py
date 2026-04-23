# =============================================================================
#  TITANIC DATASET CLEANER
#  A beginner-friendly, step-by-step data cleaning pipeline using pandas
# =============================================================================

import pandas as pd
import numpy as np
import io

# =============================================================================
# STEP 0 — Simulate the Titanic dataset (replace this block with your own CSV)
# =============================================================================
# In real usage, skip this block and just do:
#   df = pd.read_csv("titanic.csv")

RAW_CSV = """PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
2,1,1,"Cumings, Mrs. John Bradley",female,38,1,0,PC 17599,71.2833,C85,C
3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
4,1,1,"Futrelle, Mrs. Jacques Heath",female,35,1,0,113803,53.1,C123,S
5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
7,0,1,"McCarthy, Mr. Timothy J",male,54,0,0,17463,51.8625,E46,S
8,0,3,"Palsson, Master. Gosta Leonard",male,2,3,1,349909,21.075,,S
9,1,3,"Johnson, Mrs. Oscar W",female,27,0,2,347742,11.1333,,S
10,1,2,"Nasser, Mrs. Nicholas",female,14,1,0,237736,30.0708,,C
11,1,3,"Sandstrom, Miss. Marguerite Rut",female,4,1,1,PP 9549,16.7,G6,S
12,1,1,"Bonnell, Miss. Elizabeth",female,58,0,0,113783,26.55,C103,S
13,0,3,"Saundercock, Mr. William Henry",male,20,0,0,A/5. 2151,8.05,,S
14,0,3,"Andersson, Mr. Anders Johan",male,39,1,5,347082,31.275,,S
15,0,3,"Vestrom, Miss. Hulda Amanda Adolfina",female,14,0,0,350406,7.8542,,S
15,0,3,"Vestrom, Miss. Hulda Amanda Adolfina",female,14,0,0,350406,7.8542,,S
16,1,2,"Hewlett, Mrs. Mary D",female,55,0,0,248706,16,,S
17,0,3,"Rice, Master. Eugene",male,2,4,1,382652,29.125,,Q
18,1,2,"Williams, Mr. Charles Eugene",male,,0,0,244373,13,,
19,0,3,"Vander Planke, Mrs. Julius",female,31,1,0,345763,18,,S
20,1,3,"Masselmani, Mrs. Fatima",female,,0,0,2649,7.225,,C
"""

# =============================================================================
# STEP 1 — Load the Dataset
# =============================================================================
print("=" * 60)
print("STEP 1: Loading the Dataset")
print("=" * 60)

# Load from the simulated CSV string above.
# In real usage: df = pd.read_csv("titanic.csv")
df = pd.read_csv(io.StringIO(RAW_CSV))

print(f"✔  Dataset loaded successfully — {df.shape[0]} rows × {df.shape[1]} columns\n")


# =============================================================================
# STEP 2 — Explore the Dataset
# =============================================================================
print("=" * 60)
print("STEP 2: Exploring the Dataset")
print("=" * 60)

print("\n▶ Shape (rows, columns):", df.shape)

print("\n▶ Column Names:")
print(df.columns.tolist())

print("\n▶ Data Types per Column:")
print(df.dtypes)

print("\n▶ Missing Values per Column:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(missing_report[missing_report["Missing Count"] > 0])

print("\n▶ Basic Statistics (numerical columns):")
print(df.describe())


# =============================================================================
# STEP 3 — Handle Missing Values
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Handling Missing Values")
print("=" * 60)

# --- Numerical columns: fill with MEDIAN (robust to outliers) ---
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)   # pandas CoW-safe syntax
        print(f"  ✔ '{col}' — filled nulls with median ({median_val})")

# --- Categorical columns: fill with MODE (most frequent value) ---
categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]   # mode() returns a Series; [0] gets the top value
        df[col] = df[col].fillna(mode_val)     # pandas CoW-safe syntax
        print(f"  ✔ '{col}' — filled nulls with mode ('{mode_val}')")

print(f"\n  Remaining missing values after imputation: {df.isnull().sum().sum()}")


# =============================================================================
# STEP 4 — Remove Duplicate Rows
# =============================================================================
print("\n" + "=" * 60)
print("STEP 4: Removing Duplicate Rows")
print("=" * 60)

before = len(df)
df.drop_duplicates(inplace=True)
after = len(df)
removed = before - after

print(f"  Rows before: {before}")
print(f"  Rows after : {after}")
print(f"  ✔ {removed} duplicate row(s) removed.")


# =============================================================================
# STEP 5 — Convert Data Types
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Converting Data Types")
print("=" * 60)

# Survived and Pclass are integers but better treated as categories
for col in ["Survived", "Pclass"]:
    if col in df.columns:
        df[col] = df[col].astype("category")
        print(f"  ✔ '{col}' converted to category dtype")

# Sex and Embarked as category (saves memory + signals nominal data)
for col in ["Sex", "Embarked"]:
    if col in df.columns:
        df[col] = df[col].astype("category")
        print(f"  ✔ '{col}' converted to category dtype")

# Ensure Age and Fare are float
for col in ["Age", "Fare"]:
    if col in df.columns:
        df[col] = df[col].astype(float)
        print(f"  ✔ '{col}' confirmed as float64")


# =============================================================================
# STEP 6 — Rename Columns (clean, lowercase, snake_case)
# =============================================================================
print("\n" + "=" * 60)
print("STEP 6: Renaming Columns")
print("=" * 60)

rename_map = {
    "PassengerId": "passenger_id",
    "Survived":    "survived",
    "Pclass":      "ticket_class",
    "Name":        "name",
    "Sex":         "sex",
    "Age":         "age",
    "SibSp":       "siblings_spouses",
    "Parch":       "parents_children",
    "Ticket":      "ticket_number",
    "Fare":        "fare",
    "Cabin":       "cabin",
    "Embarked":    "embarkation_port",
}

df.rename(columns=rename_map, inplace=True)
print("  ✔ Renamed columns:")
for old, new in rename_map.items():
    print(f"     {old:20s} →  {new}")


# =============================================================================
# STEP 7 — Data Validation
# =============================================================================
print("\n" + "=" * 60)
print("STEP 7: Basic Data Validation")
print("=" * 60)

issues_found = 0

# Check: no remaining nulls
remaining_nulls = df.isnull().sum().sum()
if remaining_nulls == 0:
    print("  ✔ No missing values remain.")
else:
    print(f"  ✘ Warning: {remaining_nulls} missing values still present!")
    issues_found += 1

# Check: Age must be between 0 and 120
invalid_age = df[(df["age"] < 0) | (df["age"] > 120)]
if invalid_age.empty:
    print("  ✔ All age values are within valid range (0–120).")
else:
    print(f"  ✘ Warning: {len(invalid_age)} invalid age value(s) found!")
    issues_found += 1

# Check: Fare must be non-negative
invalid_fare = df[df["fare"] < 0]
if invalid_fare.empty:
    print("  ✔ All fare values are non-negative.")
else:
    print(f"  ✘ Warning: {len(invalid_fare)} negative fare value(s) found!")
    issues_found += 1

# Check: survived must be 0 or 1
invalid_survived = df[~df["survived"].isin([0, 1])]
if invalid_survived.empty:
    print("  ✔ Survived column contains only valid values (0 or 1).")
else:
    print(f"  ✘ Warning: {len(invalid_survived)} invalid survived value(s) found!")
    issues_found += 1

# Check: no duplicate passenger IDs
dup_ids = df["passenger_id"].duplicated().sum()
if dup_ids == 0:
    print("  ✔ All passenger IDs are unique.")
else:
    print(f"  ✘ Warning: {dup_ids} duplicate passenger ID(s) found!")
    issues_found += 1

print(f"\n  Validation complete — {issues_found} issue(s) detected.")


# =============================================================================
# STEP 8 — Cleaned Dataset Summary
# =============================================================================
print("\n" + "=" * 60)
print("STEP 8: Cleaned Dataset Summary")
print("=" * 60)

print(f"\n  Final shape : {df.shape[0]} rows × {df.shape[1]} columns")
print("\n  Column names and dtypes:")
print(df.dtypes.to_string())
print("\n  Descriptive statistics (numerical):")
print(df.describe(include=[np.number]).round(2))


# =============================================================================
# FINAL OUTPUT — First 5 Rows of the Cleaned Dataset
# =============================================================================
print("\n" + "=" * 60)
print("FINAL: First 5 Rows of the Cleaned Dataset")
print("=" * 60)

pd.set_option("display.max_columns", None)   # Show all columns
pd.set_option("display.width", 120)          # Wider display

print(df.head().to_string(index=False))
print("\n✅ Data cleaning pipeline complete!\n")