# Score Distribution and Behavior Analysis

### Score Distribution 
1. Wallet Score Analysis
 - ml_credit_score = ml_pred_score - min_score / (max_score - min_score) * 1000

2. Score Distribution

   Below is the distribution of predicted credit scores grouped in 100-point buckets.

    Score range   Bucket count
      
      0-99         627
      100-199       27
      200-299       25
      300-399       19
      400-499       10
      500-599       8
      600-699       4
      700-799       5
      800-899       3
      900-999      2769

3. Distribution Graph
   - Distribution of Credit Scores (0-1000).png

4. Comparsion between models for selecting best model:
   Model               MSE           R²
Random Forest       12796.354995  0.911109
Gradient Boosting   14042.272723  0.902454
Decision Tree       17421.609027  0.878980
XGBoost             16935.033203  0.882360
Ridge Regression   141947.918796  0.013950
Lasso Regression   141848.256495  0.014642
     
### Behavior Analysis

1. Wallets with **Lower Scores (0–300)**:
   - Few or no deposit transactions
   - High borrow-to-deposit ratios
   - Low or no repayment
   - Multiple liquidation calls
   - Likely risky, bot-driven, or exploit-prone behavior
     
2. Wallets with **Higher Scores (700–1000)**:
   - Made significant deposits
   - Borrowed responsibly
   - Fully repaid loans
   - Had long wallet activity spans with no liquidations
   - Reflect good DeFi behavior and reliability
3. Insights
   - Strong correlation between repay ratio and high score
   - Liquidation calls are a major negative predictor
   - Activity history also impacts the score significantly
   
   
