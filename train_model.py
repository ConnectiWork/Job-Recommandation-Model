import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Charger les données
try:
    users_df = pd.read_csv('users_synthetic.csv')
    jobs_df = pd.read_csv('jobs_synthetic.csv')
    interactions_df = pd.read_csv('interactions_synthetic.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the CSV files exist in the directory.")
    exit(1)

# Nettoyage des données
def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

users_df['skills'] = users_df['skills'].apply(safe_eval)
jobs_df['required_skills'] = jobs_df['required_skills'].apply(safe_eval)

# Gérer les valeurs nulles
users_df['rating'] = users_df['rating'].fillna(users_df['rating'].mean())
users_df['jobsCompleted'] = users_df['jobsCompleted'].fillna(0)
users_df['location'] = users_df['location'].str.capitalize()
jobs_df['location'] = jobs_df['location'].str.capitalize()

# Vérifier les doublons
users_df = users_df.drop_duplicates(subset=['id'])
jobs_df = jobs_df.drop_duplicates(subset=['job_id'])

# Normaliser jobsCompleted et duration_days
scaler = MinMaxScaler()
users_df['jobsCompleted_scaled'] = scaler.fit_transform(users_df[['jobsCompleted']])
jobs_df['duration_scaled'] = scaler.fit_transform(jobs_df[['duration_days']])
jobs_df['budget_scaled'] = scaler.fit_transform(jobs_df[['budget']])

# Préparer les features pour le content-based
users_df['skills_str'] = users_df['skills'].apply(lambda x: ' '.join(x))
jobs_df['required_skills_str'] = jobs_df['required_skills'].apply(lambda x: ' '.join(x))

tfidf = TfidfVectorizer()
user_skills_tfidf = tfidf.fit_transform(users_df['skills_str'])
job_skills_tfidf = tfidf.transform(jobs_df['required_skills_str'])
skills_similarity = cosine_similarity(user_skills_tfidf, job_skills_tfidf)

location_similarity = np.zeros((len(users_df), len(jobs_df)))
for i, user_loc in enumerate(users_df['location']):
    for j, job_loc in enumerate(jobs_df['location']):
        location_similarity[i, j] = 1 if user_loc == job_loc else 0.5

experience_similarity = np.zeros((len(users_df), len(jobs_df)))
for i, user_exp in enumerate(users_df['jobsCompleted_scaled']):
    for j, job_duration in enumerate(jobs_df['duration_scaled']):
        experience_similarity[i, j] = 1 - abs(user_exp - job_duration)

# Charger les modèles
model_files = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Random Forest': 'random_forest.pkl',
    'Gradient Boosting': 'gradient_boosting.pkl',
    'XGBoost': 'xgboost.pkl'
}
best_models = {}
for name, file in model_files.items():
    try:
        best_models[name] = joblib.load(file)
        print(f"Loaded {name} from {file}")
    except FileNotFoundError:
        print(f"Error: {file} not found. Please run the training script first.")
        exit(1)

# Fonction pour générer des recommandations avec un modèle
def recommend_jobs(model, user_id, top_n=5):
    try:
        user_idx = users_df[users_df['id'] == user_id].index[0]
    except IndexError:
        print(f"Error: User {user_id} not found in the dataset.")
        return pd.DataFrame()
    
    test_data = []
    for job_idx in range(len(jobs_df)):
        test_data.append({
            'skill_similarity': skills_similarity[user_idx, job_idx],
            'location_similarity': location_similarity[user_idx, job_idx],
            'experience_similarity': experience_similarity[user_idx, job_idx],
            'user_rating': users_df.loc[user_idx, 'rating'],
            'user_jobsCompleted': users_df.loc[user_idx, 'jobsCompleted_scaled'],
            'job_budget': jobs_df.loc[job_idx, 'budget_scaled'],
            'job_duration': jobs_df.loc[job_idx, 'duration_scaled']
        })
    
    test_df = pd.DataFrame(test_data)
    scores = model.predict_proba(test_df)[:, 1]
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommended_jobs = jobs_df.iloc[top_indices][['job_id', 'title', 'category', 'location', 'required_skills']]
    recommended_jobs['score'] = scores[top_indices]
    return recommended_jobs

# Fonction pour calculer nDCG@K
def calculate_ndcg(recommended_jobs, relevant_jobs, top_n):
    dcg = 0.0
    idcg = 0.0
    for i, job_id in enumerate(recommended_jobs['job_id'][:top_n], 1):
        if job_id in relevant_jobs:
            dcg += 1.0 / np.log2(i + 1)
        idcg += 1.0 / np.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0

# Fonction pour évaluer les recommandations
def evaluate_recommendations(model, user_id, top_n=5):
    recommended_jobs = recommend_jobs(model, user_id, top_n)
    recommended_job_ids = set(recommended_jobs['job_id'].values)
    
    relevant_jobs = set(interactions_df[
        (interactions_df['user_id'] == user_id) & 
        (interactions_df['interaction_type'] == 'applied')
    ]['job_id'].values)
    
    if len(recommended_job_ids) == 0:
        precision = 0.0
    else:
        relevant_recommended = len(recommended_job_ids.intersection(relevant_jobs))
        precision = relevant_recommended / len(recommended_job_ids)
    
    if len(relevant_jobs) == 0:
        recall = 0.0
    else:
        relevant_recommended = len(recommended_job_ids.intersection(relevant_jobs))
        recall = relevant_recommended / len(relevant_jobs)
    
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    mrr = 0.0
    for rank, job_id in enumerate(recommended_jobs['job_id'], 1):
        if job_id in relevant_jobs:
            mrr = 1.0 / rank
            break
    
    ndcg = calculate_ndcg(recommended_jobs, relevant_jobs, top_n)
    binary_accuracy = 1.0 if len(recommended_job_ids.intersection(relevant_jobs)) > 0 else 0.0
    
    return {
        'precision@{}'.format(top_n): precision,
        'recall@{}'.format(top_n): recall,
        'f1_score@{}'.format(top_n): f1_score,
        'mrr': mrr,
        'ndcg@{}'.format(top_n): ndcg,
        'binary_accuracy': binary_accuracy
    }

# Fonction pour évaluer tous les modèles
def evaluate_model(n_users=20, top_n=5):
    results = {}
    for name, model in best_models.items():
        print(f"\nEvaluating {name}...")
        metrics_list = []
        for user_id in users_df['id'].sample(n_users, random_state=42):
            recommendations = recommend_jobs(model, user_id, top_n)
            if not recommendations.empty and user_id == 'user_1':
                print(f"\nUser: {user_id}")
                print("Recommended jobs:")
                print(recommendations[['job_id', 'title', 'category', 'location', 'score']])
                print("-" * 50)
            
            metrics = evaluate_recommendations(model, user_id, top_n)
            metrics['user_id'] = user_id
            metrics_list.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        mean_metrics = metrics_df[[
            f'precision@{top_n}', 
            f'recall@{top_n}', 
            f'f1_score@{top_n}', 
            'mrr',
            f'ndcg@{top_n}',
            'binary_accuracy'
        ]].mean().to_dict()
        results[name] = mean_metrics
    
    # Comparer les résultats
    print("\nComparison of Models:")
    metrics_comparison = pd.DataFrame(results).T
    print(metrics_comparison)
    
    # Afficher le meilleur modèle
    best_model_name = metrics_comparison['precision@5'].idxmax()
    print(f"\nBest Model: {best_model_name}")
