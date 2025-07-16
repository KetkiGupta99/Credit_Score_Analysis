import json
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

with open(r"D:\Credit_Score_Project\user-wallet-transactions.json", 'r') as file:
    data = json.load(file)
    #print("JSON is properly formatted.")
#print(data)

df = pd.json_normalize(data)
df.head(5)

# Rename for clarity
df["amount"] = df["actionData.amount"].astype(float) / 1e6  # Convert to USDC
df["price_usd"] = df["actionData.assetPriceUSD"].astype(float)
df["usd_value"] = df["amount"] * df["price_usd"]

# Clean action names
df["action"] = df["action"].str.lower()

# Create boolean columns per action
df["is_deposit"] = df["action"] == "deposit"
df["is_borrow"] = df["action"] == "borrow"
df["is_repay"] = df["action"] == "repay"
df["is_liquidation"] = df["action"] == "liquidationcall"

# Transaction Type Count
sns.countplot(data=df, x="action")
plt.title("Distribution of Transaction Types")
plt.xticks(rotation=45)
plt.savefig('Distribution of Transaction Types.png') 
#plt.show()

# Correlation Heatmap
correlation = df[["amount", "price_usd", "usd_value"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig('Feature Correlation Heatmap.png')
#plt.show()

wallet_df = df.groupby("userWallet").agg({
    "usd_value": ["sum", "mean"],
    "is_deposit": "sum",
    "is_borrow": "sum",
    "is_repay": "sum",
    "is_liquidation": "sum",
    "timestamp": ["min", "max", "count"]
})

wallet_df.columns = ["_".join(col).strip() for col in wallet_df.columns.values]
wallet_df.reset_index(inplace=True)

# Create derived features
wallet_df["repay_ratio"] = wallet_df["is_repay_sum"] / wallet_df["is_borrow_sum"].replace(0, 1)
wallet_df["liquidation_rate"] = wallet_df["is_liquidation_sum"] / wallet_df["is_borrow_sum"].replace(0, 1)
wallet_df["borrow_to_deposit_ratio"] = wallet_df["is_borrow_sum"] / wallet_df["is_deposit_sum"].replace(0, 1)
wallet_df["active_days"] = (wallet_df["timestamp_max"] - wallet_df["timestamp_min"]) / (60*60*24)

# sns.histplot(wallet_df["repay_ratio"], kde=True)
# plt.title("Repayment Ratio Distribution")
# plt.savefig('Repayment Ratio Distribution.png')
#plt.show()

# sns.histplot(wallet_df["liquidation_rate"], kde=True)
# plt.title("Liquidation Rate Distribution")
# plt.savefig('Liquidation Rate Distribution.png')
#plt.show()



# wallet_df["active_days"] = (wallet_df["timestamp_max"] - wallet_df["timestamp_min"]) / (60*60*24)

# sns.histplot(wallet_df["active_days"], kde=True)
# plt.title("Wallet Activity Duration")
# plt.xlabel("Active Days")
# plt.ylabel("Wallet Count")
# plt.savefig('Wallet Activity Duration.png')
# plt.show()

# list(df.columns)

def score_wallet(row):
    score = (
        row["usd_value_sum"] * 0.1 +                 # higher deposits
        row["repay_ratio"] * 300 -                   # higher repayment ratio
        row["liquidation_rate"] * 500 +              # penalize risky behavior
        row["active_days"] * 2                       # reward longer activity
    )
    return max(0, min(1000, int(score))) 

wallet_df["credit_score"] = wallet_df.apply(score_wallet, axis=1)

features = [
    "usd_value_sum", "usd_value_mean", "is_deposit_sum",
    "is_borrow_sum", "is_repay_sum", "is_liquidation_sum",
    "repay_ratio", "liquidation_rate", "borrow_to_deposit_ratio",
    "active_days"
]

X = wallet_df[features]

# Generate synthetic target using our rule-based score
y = wallet_df["credit_score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"MSE": mse, "R²": r2}

result_df = pd.DataFrame(results).T
print(result_df)

best_model = models["Random Forest"]
wallet_df["ml_pred_score"] = best_model.predict(scaler.transform(X))

# Normalize to 0–1000
min_score = wallet_df["ml_pred_score"].min()
max_score = wallet_df["ml_pred_score"].max()

wallet_df["ml_credit_score"] = ((wallet_df["ml_pred_score"] - min_score) / (max_score - min_score) * 1000).astype(int)

wallet_scores = dict(zip(wallet_df["userWallet"], wallet_df["ml_credit_score"]))
with open("ml_wallet_scores.json", "w") as f:
    json.dump(wallet_scores, f, indent=2)

sns.histplot(wallet_df["ml_credit_score"], bins=50, kde=True)
plt.title("Distribution of ML Credit Scores")
plt.xlabel("Credit Score")
plt.ylabel("Wallet Count")
plt.savefig('Distribution of ML Credit Scores.png')
#plt.show()

# Define score bins
bins = list(range(0, 1100, 100))  
labels = [f"{i}-{i+99}" for i in bins[:-1]]  

wallet_df["score_bucket"] = pd.cut(wallet_df["credit_score"], bins=bins, labels=labels, include_lowest=True)

bucket_counts = wallet_df["score_bucket"].value_counts().sort_index()
#print(bucket_counts)

# Plot
plt.figure(figsize=(10, 5))
sns.barplot(x=bucket_counts.index, y=bucket_counts.values, hue=bucket_counts.index, palette="viridis" , legend=False)

plt.title("Distribution of Credit Scores (0-1000)")
plt.xlabel("Score Ranges")
plt.ylabel("Number of Wallets")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Distribution of Credit Scores (0-1000).png')
#plt.show()
