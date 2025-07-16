# Credit_Score_Analysis

# Objective 

- Assign each wallet a score from 0 (high risk) to 1000 (highly reliable), based solely on their transaction behavior.

#Project Structure
   - Credit_Score_Rough_Work.ipynb # Rough work 
   - credit.py/ # main python file 
   - user-wallet-transactions.json # Dataset in JSON 
   - requirements.txt # Required Python packages
   - .venv # Virtual environment 

1. Create and activate a virtual environment
   - python -m venv .venv
   - .venv\Scripts\activate

2. Install dependencies
   pip install -r requirements.txt

3. Credit Score Logic
   
  - Used two apporaches:

   ### 1. Rule-Based (Heuristic) Scoring

         This method applies weights to behavioral patterns:
      
         - Higher score for:
           - Large deposit volume
           - High repay-to-borrow ratio
           - No liquidation events
           - Long active history
      
        - Lower score for:
          - Frequent liquidation calls
          - High borrow-to-deposit ratio
          - Low repayment behaviour

  ### 2. ML-based scoring
      -  Trained several regression models:
      - Random Forest Regressor (Best performing)
      - Gradient Boosting
      - XGBoost
      - Ridge
      - Lasso
      - Decision Tree

    - Features Used:
      - Total deposit, borrow, repay (USD)
      - Borrow/deposit ration
      - Repayment ratio
      - Liquidation rate
      - Number of transactions
      - wallet activity lifespan

4. Extensibility
   Improve pipeline by:
   - Incorporating more behavioural features
   - Applying unsupervised clustering to flag abnormal wallets

5. Architecture

   Raw Aave V2 JSON User Wallet Transactions
                   |
           Flatten the data
                   |
           Clean the data
                   |
           Normalize the data
                   |
     Aggregate per Wallet (Features)
                   |
           Rule-based Score
                   |
           ML Model (Random Forest)
                   |
           Credit Score (0-1000)
                   |
               JSON Output
   
7. Processing Flow
   1. Data Load: Load raw JSON data and flatten the data into columns, Normalize the data, rename the columns name.
   2. Preprocessing:
      - Convert token amount to USD
      - Standardize actions (deposit, borrow, repay, etc.)
      - Handle missing data
   3. Feature Engineering:
      - Count actions per wallet (deposits, borrows, repays, liquidation, etc.)
      - Aggregate amounts
      - Calculate key ratios:
        - repay_ratio = is_repay_sum / is_borrow_sum
        - liquidation_rate = is_liquidation_sum / is_borrow_sum
        - borrow_to_deposit_ratio = is_borrow_sum / is_deposit_sum
        - active_days = timestamp_max - timestamp_min / (60*60*24)
   4. Modeling:
      - Train/test split
      - Model comparison using MSE, R²
      - Final model = Random Forest Regressor
  5. Scoring:
     - Predict scores using model
     - Normalize predictions to 0–1000 range
     - Output scores to ml_wallet_scores.json

  6. Analysis and Insights in analysis.md 


