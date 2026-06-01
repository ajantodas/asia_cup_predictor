import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. LOAD & CLEAN ──────────────────────────────────────
df = pd.read_csv("all_asia_cup_matches.csv")
df = df[['Team 1', 'Team 2', 'Year', 'Venue', 'Format', 'Won']].dropna()
df = df[~df['Won'].isin(['No Result', 'Tied'])]

counts = df['Won'].value_counts()
df = df[df['Won'].isin(counts[counts >= 3].index)]

print(f"Dataset: {len(df)} matches, {df['Won'].nunique()} classes")
print(df['Won'].value_counts())

# ── 2. FEATURE ENGINEERING ───────────────────────────────
strength = df['Won'].value_counts().to_dict()
df['team1_strength'] = df['Team 1'].map(strength).fillna(0)
df['team2_strength'] = df['Team 2'].map(strength).fillna(0)
df['strength_diff']  = df['team1_strength'] - df['team2_strength']

venue_team_wins = df.groupby(['Venue', 'Won']).size().reset_index(name='venue_wins')
venue_t1 = venue_team_wins.rename(columns={'Won': 'Team 1', 'venue_wins': 'venue_wins_t1'})
venue_t2 = venue_team_wins.rename(columns={'Won': 'Team 2', 'venue_wins': 'venue_wins_t2'})
df = df.merge(venue_t1, on=['Venue', 'Team 1'], how='left')
df = df.merge(venue_t2, on=['Venue', 'Team 2'], how='left')
df['venue_wins_t1'] = df['venue_wins_t1'].fillna(0)
df['venue_wins_t2'] = df['venue_wins_t2'].fillna(0)

h2h_counts = {}
for _, row in df.iterrows():
    key = (row['Team 1'], row['Team 2'])
    if key not in h2h_counts:
        h2h_counts[key] = {'t1': 0, 'total': 0}
    h2h_counts[key]['total'] += 1
    if row['Won'] == row['Team 1']:
        h2h_counts[key]['t1'] += 1

def get_h2h_rate(row):
    key  = (row['Team 1'], row['Team 2'])
    rkey = (row['Team 2'], row['Team 1'])
    if key in h2h_counts and h2h_counts[key]['total'] > 0:
        return h2h_counts[key]['t1'] / h2h_counts[key]['total']
    elif rkey in h2h_counts and h2h_counts[rkey]['total'] > 0:
        return 1 - (h2h_counts[rkey]['t1'] / h2h_counts[rkey]['total'])
    return 0.5

df['h2h_t1_win_rate'] = df.apply(get_h2h_rate, axis=1)

# ── 3. ENCODING ───────────────────────────────────────────
df_enc = pd.get_dummies(df, columns=['Team 1', 'Team 2', 'Venue', 'Format'])
X = df_enc.drop('Won', axis=1)
y = df_enc['Won']

print(f"\nFeature count: {X.shape[1]}")

# ── 4. SPLIT ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. ENSEMBLE MODEL ─────────────────────────────────────
rf = RandomForestClassifier(
    n_estimators=600, max_depth=None,
    min_samples_leaf=1, random_state=42, n_jobs=-1
)
gb = GradientBoostingClassifier(
    n_estimators=200, learning_rate=0.1,
    max_depth=4, random_state=42
)
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)], voting='soft'
)
ensemble.fit(X_train, y_train)

# ── 6. EVALUATION ─────────────────────────────────────────
y_pred = ensemble.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n=== MODEL ACCURACY: {acc:.4f} ({acc*100:.1f}%) ===")
print(classification_report(y_test, y_pred, zero_division=0))

cv = cross_val_score(rf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=42))
print(f"\nCV Score: {cv.mean():.3f} ± {cv.std():.3f}")

# ── 7. SAVE ───────────────────────────────────────────────
joblib.dump(ensemble, "asia_cup_model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\n✅ Model saved: asia_cup_model.pkl")
print("✅ Columns saved: columns.pkl")