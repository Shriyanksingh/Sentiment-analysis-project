
import pandas as pd
import numpy as np
from textblob import TextBlob
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data(path):
    df = pd.read_csv(path, parse_dates=['date'], dayfirst=False, infer_datetime_format=True)
    # Try to normalize column names
    df = df.rename(columns=lambda c: c.strip().lower())
    # Ensure required columns exist
    if 'employee_id' not in df.columns:
        if 'employee' in df.columns:
            df = df.rename(columns={'employee':'employee_id'})
        else:
            df['employee_id'] = df.index.astype(str)
    if 'message' not in df.columns and 'text' in df.columns:
        df = df.rename(columns={'text':'message'})
    if 'date' not in df.columns:
        # if no date, create a fake sequence
        df['date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    return df[['employee_id','date','message']].copy()

def label_sentiment(t):
    # TextBlob polarity: -1 .. 1
    pol = TextBlob(str(t)).sentiment.polarity
    if pol > 0.1:
        return 'Positive'
    elif pol < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def add_sentiment(df):
    df = df.copy()
    df['sentiment'] = df['message'].apply(label_sentiment)
    df['polarity'] = df['message'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df['message_length'] = df['message'].astype(str).map(len)
    df['word_count'] = df['message'].astype(str).map(lambda s: len(str(s).split()))
    return df

def monthly_scores(df):
    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')
    score_map = {'Positive':1,'Negative':-1,'Neutral':0}
    df['score'] = df['sentiment'].map(score_map)
    monthly = df.groupby(['employee_id','year_month'])['score'].sum().reset_index()
    monthly['year_month'] = monthly['year_month'].dt.to_timestamp()
    return monthly

def rankings(monthly_scores_df, month_ts):
    # month_ts should be a timestamp (start of month)
    mdf = monthly_scores_df[monthly_scores_df['year_month'] == pd.to_datetime(month_ts)]
    if mdf.empty:
        return [], []
    top_pos = mdf.sort_values(['score','employee_id'], ascending=[False, True]).head(3)
    top_neg = mdf.sort_values(['score','employee_id'], ascending=[True, True]).head(3)
    return top_pos, top_neg

def flight_risks(df):
    # Rolling 30-day count of negative mails per employee. If count >=4 at any point, flag.
    df = df.copy()
    df = df.sort_values(['employee_id','date'])
    df['is_negative'] = (df['sentiment']=='Negative').astype(int)
    flagged = set()
    for emp, g in df.groupby('employee_id'):
        dates = g[g['is_negative']==1]['date'].sort_values().reset_index(drop=True)
        # sliding window over dates
        i = 0
        j = 0
        n = len(dates)
        while i < n:
            start = dates.iloc[i]
            # move j to last index within 30 days of start
            while j < n and (dates.iloc[j] - start).days <= 30:
                j += 1
            if (j - i) >= 4:
                flagged.add(emp)
                break
            i += 1
    return sorted(list(flagged))

def build_features(monthly_df, messages_df):
    # Features at employee-month level: message_count, avg_length, negative_count, positive_count, neutral_count
    df = messages_df.copy()
    df['year_month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    agg = df.groupby(['employee_id','year_month']).agg(
        message_count = ('message','count'),
        avg_length = ('message_length','mean'),
        negative_count = (lambda x: (df.loc[x.index,'sentiment']=='Negative').sum() , 'message'),
    )
    # The above lambda aggregation is clunky; rebuild cleanly
    agg = df.groupby(['employee_id','year_month']).agg(
        message_count = ('message','count'),
        avg_length = ('message_length','mean'),
        negative_count = ('sentiment', lambda s: (s=='Negative').sum()),
        positive_count = ('sentiment', lambda s: (s=='Positive').sum()),
        neutral_count = ('sentiment', lambda s: (s=='Neutral').sum()),
    ).reset_index()
    merged = pd.merge(agg, monthly_df, left_on=['employee_id','year_month'], right_on=['employee_id','year_month'], how='left').fillna(0)
    return merged

def train_linear_model(features_df):
    df = features_df.copy().dropna()
    X = df[['message_count','avg_length','negative_count','positive_count','neutral_count']].values
    y = df['score'].values
    if len(df) < 2:
        return None, None, None, None
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    return model, mse, r2, (X_test, y_test, y_pred)

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv)>1 else 'test.csv'
    df = load_data(path)
    df = add_sentiment(df)
    df.to_csv('annotated_messages.csv', index=False)
    monthly = monthly_scores(df)
    monthly.to_csv('monthly_scores.csv', index=False)
    flags = flight_risks(df)
    print('Flight risks:', flags)
    features = build_features(monthly, df)
    features.to_csv('features_for_model.csv', index=False)
    model, mse, r2, details = train_linear_model(features)
    if model is not None:
        print('Linear Regression MSE:', mse, 'R2:', r2)
    else:
        print('Not enough data to train model.')
