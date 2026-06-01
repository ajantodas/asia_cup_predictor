from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ── Load model & data ─────────────────────────────────────
model  = joblib.load("asia_cup_model.pkl")
df_all = pd.read_csv("all_asia_cup_matches.csv")
df_all = df_all[['Team 1', 'Team 2', 'Year', 'Venue', 'Format', 'Won']].dropna()

TEAMS  = sorted(list(set(df_all['Team 1'].unique().tolist() + df_all['Team 2'].unique().tolist())))
VENUES = sorted(df_all['Venue'].unique().tolist())
FLAGS  = {
    'India': '🇮🇳', 'Pakistan': '🇵🇰', 'Sri Lanka': '🇱🇰',
    'Bangladesh': '🇧🇩', 'Afghanistan': '🇦🇫', 'Hong Kong': '🇭🇰',
    'United Arab Emirates': '🇦🇪'
}

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html', teams=TEAMS, venues=VENUES, flags=FLAGS)


@app.route('/predict', methods=['POST'])
def predict():
    data         = request.get_json()
    team1        = data.get('team1')
    team2        = data.get('team2')
    venue        = data.get('venue')
    match_format = data.get('format', 'ODI')
    year         = int(data.get('year', 2025))

    try:
        input_df = pd.DataFrame(columns=model.feature_names_in_)
        input_df.loc[0] = 0
        input_df['Year'] = year

        strength = df_all['Won'].value_counts().to_dict()
        input_df['team1_strength'] = strength.get(team1, 0)
        input_df['team2_strength'] = strength.get(team2, 0)
        input_df['strength_diff']  = input_df['team1_strength'] - input_df['team2_strength']

        for col in [f'Team 1_{team1}', f'Team 2_{team2}',
                    f'Venue_{venue}', f'Format_{match_format}']:
            if col in input_df.columns:
                input_df[col] = 1

        prediction = model.predict(input_df)[0]

        try:
            proba     = model.predict_proba(input_df)[0]
            classes   = model.classes_
            prob_dict = dict(zip(classes, proba))
            p1 = round(prob_dict.get(team1, 0.5) * 100, 1)
            p2 = round(prob_dict.get(team2, 0.5) * 100, 1)
        except Exception:
            p1, p2 = (60.0, 40.0) if prediction == team1 else (40.0, 60.0)

        h2h = df_all[
            ((df_all['Team 1'] == team1) & (df_all['Team 2'] == team2)) |
            ((df_all['Team 1'] == team2) & (df_all['Team 2'] == team1))
        ].to_dict(orient='records')

        return jsonify({
            'success':      True,
            'winner':       prediction,
            'team1_prob':   p1,
            'team2_prob':   p2,
            'h2h_count':    len(h2h),
            'h2h_t1_wins':  sum(1 for m in h2h if m['Won'] == team1),
            'h2h_t2_wins':  sum(1 for m in h2h if m['Won'] == team2),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/stats')
def stats():
    wins = df_all['Won'].value_counts().to_dict()
    return jsonify({
        'total_matches': len(df_all),
        'teams':         TEAMS,
        'overall_wins':  wins,
    })


@app.route('/h2h')
def h2h():
    t1 = request.args.get('t1')
    t2 = request.args.get('t2')
    matches = df_all[
        ((df_all['Team 1'] == t1) & (df_all['Team 2'] == t2)) |
        ((df_all['Team 1'] == t2) & (df_all['Team 2'] == t1))
    ].sort_values('Year', ascending=False).to_dict(orient='records')
    return jsonify(matches)


@app.route('/venue-stats')
def venue_stats():
    venue = request.args.get('venue')
    vm    = df_all[df_all['Venue'] == venue]
    wins  = vm['Won'].value_counts().to_dict()
    return jsonify({'venue': venue, 'total': len(vm), 'wins': wins})


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)