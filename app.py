import flask
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = flask.Flask(__name__, template_folder='templates')

df = pd.read_excel('./data/Meals_w_Goals_deid_snapshot.xlsx')

df.drop(df[df.preferred_locale == 'es'].index, inplace=True)
cleaned_df = df.drop(columns=['goal_id','expert_explanation', 'meal_type', 'goal_short_name', 'expert_assessment', 'user_id', 'meal_id', 'goal_id', 'expert_assessment', 'carbs_grams', 'protein_grams','fat_grams', 'fiber_grams', 'calories', 'carbs_RD_explanation', 'protein_RD_explanation', 'fat_RD_explanation', 'fiber_RD_explanation', 'calories_RD_explanation', 'preferred_locale'])
cleaned_df['meal_title'] = cleaned_df['meal_title'].drop_duplicates()

cleaned_df = cleaned_df[cleaned_df['meal_title'].notna()]

cleaned_df["meal_ingredients"] = cleaned_df["meal_ingredients"].astype(str)


tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3),
                      stop_words='english')

# Filling NaNs with empty string
cleaned_df["meal_ingredients"] = cleaned_df["meal_ingredients"].fillna('')
tfv_matrix = tfv.fit_transform(cleaned_df["meal_ingredients"])
sig = cosine_similarity(tfv_matrix, tfv_matrix)
# Reverse mapping of indices and titles
indices = pd.Series(cleaned_df.index, index=cleaned_df['meal_title']).drop_duplicates()
indices = indices.drop_duplicates()

def get_recommendations(title, sig=sig):
    # Get the index corresponding to meal_title
    idx = indices[title]

    # Get the pairwsie similarity scores
    sig_scores = list(enumerate(sig[idx]))

    # Sort the meal_tile
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

    # Scores of the 5 most similar meals
    sig_scores = sig_scores[1:6]

    # meal_title indices
    meal_indices = [i[0] for i in sig_scores]

    # Top 5 most similar meals
    return cleaned_df['meal_title'].iloc[meal_indices]

@app.route('/', methods=['GET', 'POST'])

def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))

    if flask.request.method == 'POST':
        m_name = flask.request.form['meal_name']
        # m_name = m_name.title()
        result_final = get_recommendations(m_name)
        names = []
        for i in range(len(result_final)):
            names.append(result_final.iloc[i])
        print(names)
        return(flask.render_template('positive.html', meal_names = names))

        # return flask.render_template('positive.html', meal_names = names)

if __name__ == '__main__':
    app.run(debug=True)

# host='0.0.0.0',port=5000,debug=True,use_reloader=True

