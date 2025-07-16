# Score Distribution and Behavior Analysis

### Score Distribution 
1. Wallet Score Analysis
 - ml_credit_score = ml_pred_score - min_score / (max_score - min_score) * 1000

2. Score Distribution

   Below is the distribution of predicted credit scores grouped in 100-point buckets.

| Score Range | Wallet Count |
|-------------|--------------|
| 0–100       | 58           |
| 100–200     | 134          |
| 200–300     | 295          |
| 300–400     | 680          |
| 400–500     | 950          |
| 500–600     | 720          |
| 600–700     | 580          |
| 700–800     | 390          |
| 800–900     | 160          |
| 900–1000    | 33           |

3. Distribution Graph
   - Distribution of Credit Scores (0-1000).png
     
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
   
   
