    # Employee Sentiment Analysis (TextBlob implementation)

    This repository contains a self-contained pipeline that performs:

    - Sentiment labeling (Positive / Negative / Neutral) using **TextBlob**.
    - Exploratory Data Analysis and visualizations (notebook).
    - Monthly sentiment scoring per employee.
    - Employee ranking (top 3 positive & negative per month).
    - Flight risk identification (>=4 negative messages in any rolling 30-day window).
    - Simple linear regression model for sentiment trends with basic features.

    ## Files
    - `test.csv` - Input data (converted from uploaded `test.xlsx` if provided).
    - `sentiment_analysis.py` - Main Python script implementing the pipeline.
    - `main.ipynb` - Notebook demonstrating usage and outputs.
    - `annotated_messages.csv` - Generated file with sentiment labels (after running).
    - `monthly_scores.csv` - Monthly aggregated scores per employee.
    - `features_for_model.csv` - Features used for linear regression.
    - `README.md` - This file.
    - `.env.example` - Example environment variables file.
    - `requirements.txt` - Python dependencies.

    ## How to run (locally)
    1. Create a virtual environment and activate it.
    2. Install dependencies: `pip install -r requirements.txt`
    3. Place your `test.csv` (or `test.xlsx`) in the project folder.
    4. Run the pipeline: `python3 sentiment_analysis.py test.csv`
    5. Or open `main.ipynb` and run the notebook cells.

    ## Sentiment labeling approach
    We use **TextBlob** polarity scores:
    - polarity > 0.1 => **Positive**
    - polarity < -0.1 => **Negative**
    - otherwise => **Neutral**

    This thresholding provides a reproducible, simple rule that works reasonably for short messages. Replace the `label_sentiment` function in `sentiment_analysis.py` to use an LLM or another classifier if higher accuracy is required.

    ## Flight risk rule
    An employee is flagged as flight risk if they have **4 or more negative messages within any rolling 30-day window**.

    ## Predictive model
    A simple linear regression model is trained at the employee-month level using features:
    - message_count
- avg_length
- negative_count
- positive_count
- neutral_count

Model metrics printed by the script include MSE and R² (if enough data exists).

    ## Notes
    - The included notebook runs the script and displays sample outputs.
    - For production, consider replacing TextBlob with a fine-tuned classifier or an LLM for improved sentiment accuracy.
