from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import joblib

# ── Load ──────────────────────────────────────────────────────────────────────
train = pd.read_csv('fraudTrain.csv', index_col=0)
test  = pd.read_csv('fraudTest.csv',  index_col=0)

# ── Feature engineering ───────────────────────────────────────────────────────
def engineer_features(df):
    dt = pd.to_datetime(df['trans_date_trans_time'])
    df['time_of_day'] = dt.dt.hour
    df['day_of_week'] = dt.dt.dayofweek
    df['day_of_month'] = dt.dt.day
    df['month']       = dt.dt.month
    df['age']         = (pd.Timestamp.today() - pd.to_datetime(df['dob'])).dt.days // 365

    def add_velocity_features(df, cc_col='cc_num',
                              time_col='trans_date_trans_time',
                              amt_col='amt'):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([cc_col, time_col]).reset_index(drop=True)

        df['tx_count_1h'] = (
            df.groupby(cc_col)[time_col]
            .transform(lambda x: x.expanding().count()
                                 - x.shift(1).fillna(x.iloc[0])
                       .transform(lambda _: 0))
        )

        # cleaner approach: use a time-indexed rolling per group
        results = []
        for card, group in df.groupby(cc_col):
            group = group.set_index(time_col)
            group['tx_count_1h'] = group[amt_col].rolling('1h').count()
            group['tx_count_24h'] = group[amt_col].rolling('24h').count()
            group['amt_sum_24h'] = group[amt_col].rolling('24h').sum()

            results.append(group.reset_index())

        return pd.concat(results).sort_index()

    return add_velocity_features(df)

train = engineer_features(train)
test  = engineer_features(test)

# ── OHE category ──────────────────────────────────────────────────────────────
ohe = OneHotEncoder(sparse_output=False, dtype=int, handle_unknown='ignore')
ohe.fit(train[['category']])  # fit on train only
joblib.dump(ohe, 'ohe_encoder.pkl')
cat_cols = ohe.get_feature_names_out(['category'])

train_ohe = pd.DataFrame(ohe.transform(train[['category']]), columns=cat_cols, index=train.index)
test_ohe  = pd.DataFrame(ohe.transform(test[['category']]),  columns=cat_cols, index=test.index)

train = pd.concat([train, train_ohe], axis=1)
test  = pd.concat([test,  test_ohe],  axis=1)

# ── Drop columns ──────────────────────────────────────────────────────────────
DROP = [
    'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
    'street', 'city', 'state', 'zip', 'lat', 'long', 'merch_lat', 'merch_long',
    'city_pop', 'job', 'dob', 'trans_num', 'unix_time', 'gender', 'category'
]
train = train.drop(columns=DROP)
test  = test.drop(columns=DROP)

# ── Split features / target ───────────────────────────────────────────────────
X_train = train.drop(columns=['is_fraud'])
y_train = train['is_fraud']
X_test  = test.drop(columns=['is_fraud'])
y_test  = test['is_fraud']

# ── Save ──────────────────────────────────────────────────────────────────────
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv',   index=False)
y_test.to_csv('y_test.csv',   index=False)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Columns: {list(X_train.columns)}")