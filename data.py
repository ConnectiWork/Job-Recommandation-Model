import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

# Initialisation de Faker pour des données en français
fake = Faker('fr_FR')

# Paramètres
n_users = 500  # Nombre d'utilisateurs
n_jobs = 1000  # Nombre de jobs
n_interactions = 2000  # Nombre d'interactions (candidatures, jobs sauvegardés)

# Listes pour personnaliser les données
skills_list = ['Python', 'JavaScript', 'UI/UX', 'SQL', 'Django', 'React', 'Figma', 'Marketing Digital', 'Gestion de Projet', 'Cybersécurité', 'Traduction', 'Logistique']
locations = ['Tanger', 'Casablanca', 'Rabat', 'Marrakech', 'Fès', 'Agadir']
languages_list = ['Français', 'Anglais', 'Arabe', 'Espagnol']
categories = ['Développement Web', 'Design', 'Data Science', 'Marketing', 'Événementiel', 'Logistique', 'Cybersécurité']
professions = ['Développeur', 'Designer', 'Data Scientist', 'Marketeur', 'Manager de Projet', 'Traducteur', 'Logisticien']

# Générer les utilisateurs
users = []
for i in range(n_users):
    created_at = fake.date_time_between(start_date='-2y', end_date='now')
    user = {
        'id': f'user_{i+1}',
        'email': fake.email(),
        'fullName': fake.name(),
        'photoUrl': fake.image_url() if random.random() > 0.5 else None,
        'phoneNumber': fake.phone_number() if random.random() > 0.7 else None,
        'bio': fake.text(max_nb_chars=100) if random.random() > 0.6 else None,
        'location': random.choice(locations),
        'profession': random.choice(professions) if random.random() > 0.4 else None,
        'skills': random.sample(skills_list, random.randint(2, 6)),
        'languages': random.sample(languages_list, random.randint(1, 3)),
        'rating': round(random.uniform(3.0, 5.0), 1) if random.random() > 0.3 else None,
        'jobsCompleted': random.randint(0, 20) if random.random() > 0.2 else None,
        'jobsPosted': random.randint(0, 10) if random.random() > 0.5 else None,
        'isEmployer': random.random() > 0.7,
        'isWorker': random.random() > 0.3,
        'createdAt': created_at,
        'lastActive': created_at + timedelta(days=random.randint(1, 700)) if random.random() > 0.4 else None,
        'preferences': {'preferred_category': random.choice(categories), 'min_budget': random.randint(500, 3000)} if random.random() > 0.5 else None,
        'settings': {'notifications': random.choice(['email', 'push', 'none'])} if random.random() > 0.5 else None,
        'savedJobs': [],
        'appliedJobs': [],
        'postedJobs': [],
        'connections': []
    }
    users.append(user)

# Générer les jobs
jobs = []
for i in range(n_jobs):
    job = {
        'job_id': f'job_{i+1}',
        'title': fake.job(),
        'category': random.choice(categories),
        'required_skills': random.sample(skills_list, random.randint(2, 5)),
        'location': random.choice(locations),
        'budget': random.randint(500, 5000),
        'duration_days': random.randint(5, 90),
        'description': fake.text(max_nb_chars=200),
        'posted_by': random.choice([u['id'] for u in users if u['isEmployer']]),
        'created_at': fake.date_time_between(start_date='-1y', end_date='now')
    }
    jobs.append(job)

# Générer des interactions (candidatures et jobs sauvegardés)
interactions = []
for _ in range(n_interactions):
    user = random.choice([u for u in users if u['isWorker']])
    job = random.choice(jobs)
    interaction_type = random.choice(['applied', 'saved'])
    if interaction_type == 'applied':
        user['appliedJobs'].append(job['job_id'])
    else:
        user['savedJobs'].append(job['job_id'])
    interactions.append({
        'user_id': user['id'],
        'job_id': job['job_id'],
        'interaction_type': interaction_type,
        'timestamp': fake.date_time_between(start_date='-1y', end_date='now')
    })

# Convertir en DataFrames
users_df = pd.DataFrame(users)
jobs_df = pd.DataFrame(jobs)
interactions_df = pd.DataFrame(interactions)

# Sauvegarder en CSV
users_df.to_csv('users_synthetic.csv', index=False)
jobs_df.to_csv('jobs_synthetic.csv', index=False)
interactions_df.to_csv('interactions_synthetic.csv', index=False)

print("Données synthétiques générées : users_synthetic.csv, jobs_synthetic.csv, interactions_synthetic.csv")