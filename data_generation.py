# Génération de données synthétiques pour système de recommandation d'emplois
# --------------------------------------------------------------------------
# Ce notebook génère des données synthétiques adaptées au Maroc et à divers domaines professionnels

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration pour la reproductibilité
np.random.seed(42)
random.seed(42)

# 1. Définition des attributs possibles
# -------------------------------------

# Villes du Maroc
locations = [
    'Casablanca', 'Rabat', 'Marrakech', 'Fès', 'Tanger', 'Agadir', 'Meknès', 
    'Oujda', 'Kénitra', 'Tétouan', 'El Jadida', 'Safi', 'Mohammedia', 'Khouribga',
    'Béni Mellal', 'Nador', 'Taza', 'Settat', 'Berrechid', 'Remote (Maroc)'
]

# Domaines professionnels plus variés
domains = [
    'IT & Développement', 'Marketing Digital', 'Design & Création', 'Rédaction & Traduction',
    'Ingénierie', 'Finance & Comptabilité', 'Droit & Juridique', 'Ressources Humaines',
    'Éducation & Formation', 'Santé', 'Tourisme & Hôtellerie', 'Commerce & Vente', 
    'Communication', 'Architecture', 'Consulting', 'BTP & Construction'
]

# Compétences par domaine
skills_by_domain = {
    'IT & Développement': [
        'JavaScript', 'Python', 'Java', 'PHP', 'C#', 'C++', 'SQL', 'React', 'Angular', 
        'Vue.js', 'Node.js', 'Django', 'Laravel', 'Flutter', 'Swift', 'Kotlin', 
        'WordPress', 'Shopify', 'AWS', 'Azure', 'DevOps', 'Docker', 'Kubernetes',
        'Machine Learning', 'Data Science', 'Blockchain', 'Cybersécurité', 'UI/UX',
        'SEO', 'Mobile Development', 'Game Development', 'QA & Testing'
    ],
    'Marketing Digital': [
        'SEO', 'SEM', 'SMM', 'Content Marketing', 'Email Marketing', 'Copywriting',
        'Google Ads', 'Facebook Ads', 'Instagram Ads', 'TikTok Ads', 'Analytics',
        'Marketing Strategy', 'Branding', 'CRM', 'Marketing Automation', 'Growth Hacking',
        'Affiliate Marketing', 'Lead Generation', 'Conversion Optimization', 'UX Research'
    ],
    'Design & Création': [
        'Photoshop', 'Illustrator', 'InDesign', 'After Effects', 'Premiere Pro', 'Figma',
        'Sketch', 'XD', 'UI Design', 'UX Design', 'Motion Design', 'Graphic Design',
        'Web Design', 'Logo Design', '3D Modeling', 'Animation', 'Branding',
        'Typography', 'Product Design', 'Packaging Design', 'Illustration'
    ],
    'Rédaction & Traduction': [
        'Copywriting', 'Content Writing', 'Technical Writing', 'SEO Writing', 
        'Academic Writing', 'Ghostwriting', 'Editing', 'Proofreading',
        'French Translation', 'English Translation', 'Arabic Translation', 
        'Spanish Translation', 'German Translation', 'Subtitling', 'Transcription',
        'Creative Writing', 'Journalistic Writing', 'Legal Writing'
    ],
    'Ingénierie': [
        'Génie Civil', 'Génie Électrique', 'Génie Mécanique', 'Génie Industriel',
        'Génie Informatique', 'Génie des Procédés', 'Génie Énergétique',
        'AutoCAD', 'SolidWorks', 'MATLAB', 'Simulation', 'Prototypage',
        'Gestion de Projet', 'Assurance Qualité', 'IoT', 'Robotique',
        'Conception Technique', 'Développement Durable'
    ],
    'Finance & Comptabilité': [
        'Comptabilité Générale', 'Comptabilité Analytique', 'Audit', 'Fiscalité',
        'Contrôle de Gestion', 'Analyse Financière', 'Trésorerie', 'Budgétisation',
        'ERP Finance', 'SAP', 'Sage', 'Excel Avancé', 'Reporting Financier',
        'IFRS', 'Gestion des Risques', 'Consolidation', 'Finance d\'Entreprise'
    ],
    'Droit & Juridique': [
        'Droit Commercial', 'Droit Civil', 'Droit du Travail', 'Droit Fiscal',
        'Droit des Affaires', 'Droit Immobilier', 'Droit International',
        'Droit des Contrats', 'Droit des Sociétés', 'Propriété Intellectuelle',
        'RGPD', 'Contentieux', 'Médiation', 'Arbitrage', 'Conformité'
    ],
    'Ressources Humaines': [
        'Recrutement', 'Formation', 'Gestion des Talents', 'SIRH', 'Paie',
        'Administration du Personnel', 'Développement RH', 'Relations Sociales',
        'Gestion des Compétences', 'Assessment', 'Onboarding', 'GPEC',
        'Management RH', 'Mobilité Interne', 'Bien-être au Travail'
    ],
    'Éducation & Formation': [
        'E-learning', 'LMS', 'Instructional Design', 'Pédagogie', 'Tutorat',
        'Création de Cours', 'Formation Professionnelle', 'Coaching',
        'Évaluation', 'Gestion de Formation', 'Blended Learning',
        'Sciences de l\'Éducation', 'MOOC', 'Édition Pédagogique'
    ],
    'Santé': [
        'Télémédecine', 'Dossier Patient', 'Pharmacovigilance', 'Recherche Clinique',
        'Santé Publique', 'Qualité Sanitaire', 'Réglementation Médicale',
        'Medical Writing', 'Biologie Médicale', 'Imagerie Médicale',
        'Gestion Hospitalière', 'Conseil Médical', 'Santé Digitale'
    ],
    'Tourisme & Hôtellerie': [
        'Gestion Hôtelière', 'Revenue Management', 'Réservation', 'CRM Tourisme',
        'Marketing Touristique', 'Organisation d\'Événements', 'Gestion de Restaurant',
        'Conciergerie', 'Tour Operating', 'Gestion de Destination',
        'Hospitality Management', 'Expérience Client'
    ],
    'Commerce & Vente': [
        'Vente B2B', 'Vente B2C', 'Négociation', 'Account Management', 'CRM',
        'Sales Development', 'Business Development', 'Export', 'E-commerce',
        'Retail', 'Category Management', 'Merchandising', 'Gestion Commerciale',
        'Sales Enablement', 'Inside Sales', 'Field Sales'
    ],
    'Communication': [
        'Relations Presse', 'Relations Publiques', 'Communication Interne',
        'Communication Corporate', 'Communication de Crise', 'Communication Digitale',
        'Community Management', 'Event Management', 'Stratégie de Communication',
        'Sponsoring', 'Media Planning', 'Communication Politique'
    ],
    'Architecture': [
        'AutoCAD', 'Revit', 'SketchUp', 'BIM', 'ArchiCAD', 'Rhinoceros 3D',
        'Architecture d\'Intérieur', 'Aménagement Urbain', 'Conception Architecturale',
        'Rendu 3D', 'Design Durable', 'Rénovation', 'Architecture Paysagère'
    ],
    'Consulting': [
        'Stratégie d\'Entreprise', 'Conseil en Management', 'Conseil en Organisation',
        'Transformation Digitale', 'Change Management', 'Process Optimization',
        'Business Analysis', 'Due Diligence', 'Benchmark', 'KPI', 'Lean Management',
        'Six Sigma', 'Conseil en Innovation', 'Études de Marché'
    ],
    'BTP & Construction': [
        'Gestion de Chantier', 'Maîtrise d\'Œuvre', 'Maîtrise d\'Ouvrage',
        'Étude de Prix', 'Planning', 'BIM', 'AutoCAD', 'Sécurité Chantier',
        'Coordination Travaux', 'Génie Civil', 'Bureau d\'Études',
        'Architecture', 'Immobilier', 'Construction Durable'
    ]
}

# Professions par domaine
professions_by_domain = {
    'IT & Développement': [
        'Développeur Frontend', 'Développeur Backend', 'Développeur Fullstack', 
        'Développeur Mobile', 'Data Scientist', 'DevOps Engineer', 'Cloud Architect',
        'Product Manager', 'Scrum Master', 'CTO', 'UI/UX Designer', 'Admin Sys',
        'Data Engineer', 'QA Engineer', 'Security Engineer'
    ],
    'Marketing Digital': [
        'Digital Marketer', 'SEO Specialist', 'Content Manager', 'Social Media Manager',
        'Growth Hacker', 'Performance Marketing Manager', 'Analytics Manager',
        'Email Marketing Specialist', 'SEM Expert', 'Community Manager',
        'Brand Manager', 'Marketing Automation Expert'
    ],
    'Design & Création': [
        'Graphic Designer', 'UI Designer', 'UX Designer', 'Motion Designer',
        'Art Director', 'Creative Director', 'Web Designer', 'Product Designer',
        '3D Artist', 'Illustrator', 'Logo Designer', 'Visual Designer'
    ],
    'Rédaction & Traduction': [
        'Copywriter', 'Content Writer', 'Traducteur', 'Rédacteur Web', 
        'Journaliste', 'Technical Writer', 'Correcteur', 'Transcripteur',
        'Editor', 'Ghostwriter', 'Rédacteur SEO', 'Concepteur-Rédacteur'
    ],
    'Ingénierie': [
        'Ingénieur Civil', 'Ingénieur Électrique', 'Ingénieur Mécanique',
        'Ingénieur R&D', 'Ingénieur Qualité', 'Ingénieur Projet',
        'Ingénieur Process', 'Ingénieur Industriel', 'Chef de Projet Technique',
        'Directeur Technique', 'Ingénieur d\'Études'
    ],
    'Finance & Comptabilité': [
        'Comptable', 'Contrôleur de Gestion', 'Directeur Financier', 'Auditeur',
        'Expert Comptable', 'Analyste Financier', 'Trésorier', 'Credit Manager',
        'Responsable Fiscal', 'Consolideur', 'Financial Controller'
    ],
    'Droit & Juridique': [
        'Juriste d\'Entreprise', 'Avocat', 'Consultant Juridique', 'Fiscaliste',
        'Juriste Droit Social', 'Juriste Propriété Intellectuelle',
        'Contract Manager', 'Compliance Officer', 'Paralegal', 'Legal Counsel'
    ],
    'Ressources Humaines': [
        'Responsable RH', 'Chargé de Recrutement', 'HRBP', 'Responsable Formation',
        'Talent Acquisition Manager', 'Responsable Paie', 'DRH',
        'Responsable Mobilité', 'Consultant RH', 'Responsable SIRH'
    ],
    'Éducation & Formation': [
        'Formateur', 'Concepteur Pédagogique', 'Responsable Formation',
        'Coach Professionnel', 'Enseignant', 'Tuteur E-learning',
        'Conseiller Pédagogique', 'Directeur de Formation',
        'Ingénieur Pédagogique', 'Consultant en Formation'
    ],
    'Santé': [
        'Médecin', 'Infirmier', 'Pharmacien', 'Chercheur Clinique',
        'Responsable Qualité Sanitaire', 'Consultant Medical',
        'Responsable Affaires Réglementaires', 'Medical Writer',
        'Directeur d\'Établissement de Santé', 'Ingénieur Biomédical'
    ],
    'Tourisme & Hôtellerie': [
        'Revenue Manager', 'Directeur Hôtelier', 'Chef de Réception',
        'Responsable Événementiel', 'Tour Operator', 'Directeur de Restaurant',
        'Concierge', 'Chef de Produit Touristique', 'Destination Manager',
        'Travel Planner', 'Hospitality Manager'
    ],
    'Commerce & Vente': [
        'Commercial B2B', 'Commercial B2C', 'Account Manager', 'Sales Manager',
        'Business Developer', 'Responsable E-commerce', 'Category Manager',
        'Responsable Commercial', 'Key Account Manager', 'Sales Director',
        'Export Manager', 'Responsable Merchandising'
    ],
    'Communication': [
        'Chargé de Communication', 'Responsable RP', 'Community Manager',
        'Responsable Communication Interne', 'Directeur de Communication',
        'Attaché de Presse', 'Event Manager', 'Responsable Communication Digitale',
        'Media Planner', 'Brand Content Manager'
    ],
    'Architecture': [
        'Architecte', 'Architecte d\'Intérieur', 'Designer d\'Espace',
        'BIM Manager', 'Architecte Paysagiste', 'Chef de Projet Architecture',
        'Responsable Bureau d\'Études', 'Urban Designer', 'Dessinateur Projeteur',
        'Directeur de Projet Architectural'
    ],
    'Consulting': [
        'Consultant en Management', 'Consultant en Stratégie', 'Business Analyst',
        'Consultant Transformation', 'Chef de Projet Conseil', 'Consultant Digital',
        'Consultant Opérations', 'Consultant RH', 'Consultant Change Management',
        'Senior Consultant', 'Manager Consulting'
    ],
    'BTP & Construction': [
        'Chef de Chantier', 'Conducteur de Travaux', 'Directeur de Travaux',
        'Maître d\'Œuvre', 'Chargé d\'Affaires BTP', 'Ingénieur Travaux',
        'Responsable Bureau d\'Études', 'BIM Coordinator', 'Économiste de la Construction',
        'Architecte', 'Directeur de Programme Immobilier'
    ]
}

# Langues
languages = ['Français', 'Arabe', 'Anglais', 'Espagnol', 'Allemand', 'Italien', 'Chinois']

# Types d'emploi et durées
job_types = ['Temps plein', 'Temps partiel', 'Contrat', 'Freelance', 'Mission ponctuelle']
job_durations = ['1 semaine', '2 semaines', '1 mois', '3 mois', '6 mois', '1 an']

# 2. Fonction de génération des utilisateurs
# -----------------------------------------

def generate_users(num_users=1000):
    users = []
    
    for i in range(num_users):
        user_id = f"user_{i}"
        
        # Sélectionner un domaine aléatoire pour l'utilisateur
        user_domain = random.choice(domains)
        
        # Sélectionner une profession dans ce domaine
        user_profession = random.choice(professions_by_domain[user_domain])
        
        # Sélectionner des compétences principalement de ce domaine
        domain_skills = skills_by_domain[user_domain]
        num_domain_skills = random.randint(3, min(8, len(domain_skills)))
        user_skills = random.sample(domain_skills, num_domain_skills)
        
        # Ajouter quelques compétences d'autres domaines (interdisciplinarité)
        other_domains = [d for d in domains if d != user_domain]
        if other_domains and random.random() < 0.7:  # 70% de chance d'avoir des compétences interdisciplinaires
            num_other_domains = random.randint(1, min(3, len(other_domains)))
            selected_other_domains = random.sample(other_domains, num_other_domains)
            
            for other_domain in selected_other_domains:
                other_domain_skills = skills_by_domain[other_domain]
                num_other_skills = random.randint(1, 3)
                other_skills = random.sample(other_domain_skills, min(num_other_skills, len(other_domain_skills)))
                user_skills.extend(other_skills)
        
        # Langues
        user_languages = random.sample(languages, random.randint(1, 3))
        if 'Arabe' not in user_languages and random.random() < 0.8:  # Majorité parle arabe au Maroc
            user_languages.append('Arabe')
        if 'Français' not in user_languages and random.random() < 0.7:  # Beaucoup parlent français au Maroc
            user_languages.append('Français')
        
        # Associer un niveau de compétence à chaque skill
        skill_levels = {}
        for skill in user_skills:
            # Niveau entre 1 et 5
            skill_levels[skill] = random.randint(1, 5)
        
        # Générer des expériences passées
        num_experiences = random.randint(0, 5)
        experiences = []
        for j in range(num_experiences):
            exp_domain = user_domain if random.random() < 0.8 else random.choice(domains)
            exp_skills = random.sample(user_skills, random.randint(1, min(3, len(user_skills))))
            
            exp = {
                'title': f"{random.choice(professions_by_domain[exp_domain])} - Projet {j}",
                'domain': exp_domain,
                'duration': random.choice(job_durations),
                'skills_used': exp_skills,
                'date': (datetime.now() - timedelta(days=random.randint(30, 1000))).strftime('%Y-%m-%d')
            }
            experiences.append(exp)
        
        user = {
            'id': user_id,
            'fullName': f"Utilisateur {i}",
            'email': f"user{i}@example.com",
            'location': random.choice(locations),
            'domain': user_domain,
            'profession': user_profession,
            'skills': user_skills,
            'skill_levels': skill_levels,
            'languages': user_languages,
            'rating': round(random.uniform(3.0, 5.0), 1),
            'jobsCompleted': random.randint(0, 50),
            'experiences': experiences,
            'isWorker': True,
            'lastActive': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
        }
        users.append(user)
    
    return pd.DataFrame(users)

# 3. Fonction de génération des offres d'emploi
# --------------------------------------------

def generate_jobs(num_jobs=2000):
    jobs = []
    
    for i in range(num_jobs):
        job_id = f"job_{i}"
        
        # Sélectionner un domaine aléatoire pour l'offre
        job_domain = random.choice(domains)
        
        # Sélectionner une profession dans ce domaine
        job_profession = random.choice(professions_by_domain[job_domain])
        
        # Sélectionner des compétences requises pour ce domaine
        domain_skills = skills_by_domain[job_domain]
        num_domain_skills = random.randint(2, min(5, len(domain_skills)))
        required_skills = random.sample(domain_skills, num_domain_skills)
        
        # Eventuellement ajouter quelques compétences d'autres domaines
        if random.random() < 0.3:  # 30% de chance d'avoir des compétences d'autres domaines
            other_domains = [d for d in domains if d != job_domain]
            if other_domains:
                other_domain = random.choice(other_domains)
                other_domain_skills = skills_by_domain[other_domain]
                num_other_skills = random.randint(1, 2)
                other_skills = random.sample(other_domain_skills, min(num_other_skills, len(other_domain_skills)))
                required_skills.extend(other_skills)
        
        # Niveau de compétence requis pour chaque skill
        skill_requirements = {}
        for skill in required_skills:
            skill_requirements[skill] = random.randint(1, 5)
        
        # Générer un titre d'emploi significatif
        job_title = f"{job_profession} - {random.choice(['Projet', 'Mission', 'Poste'])} {job_domain}"
        
        # Langues requises
        required_languages = ['Arabe'] if random.random() < 0.8 else []
        if random.random() < 0.7:
            required_languages.append('Français')
        if random.random() < 0.5:
            required_languages.append('Anglais')
        # Éviter les doublons
        required_languages = list(set(required_languages))
        
        # Générer les informations de l'emploi
        job = {
            'id': job_id,
            'title': job_title,
            'description': f"Description de l'offre d'emploi {i} dans le domaine {job_domain}",
            'location': random.choice(locations),
            'domain': job_domain,
            'required_skills': required_skills,
            'skill_requirements': skill_requirements,
            'job_type': random.choice(job_types),
            'duration': random.choice(job_durations),
            'required_languages': required_languages,
            'posted_date': (datetime.now() - timedelta(days=random.randint(0, 60))).strftime('%Y-%m-%d'),
            'salary_range': f"{random.randint(300, 800)}-{random.randint(800, 1500)} MAD/jour",
            'employer_id': f"employer_{random.randint(0, 100)}"
        }
        jobs.append(job)
    
    return pd.DataFrame(jobs)

# 4. Générer les interactions entre utilisateurs et emplois
# -------------------------------------------------------

def generate_interactions(users_df, jobs_df, interaction_rate=0.05):
    interactions = []
    
    for _, user in users_df.iterrows():
        # Définir des emplois pertinents pour cet utilisateur basés sur ses compétences et son domaine
        relevant_jobs = []
        for _, job in jobs_df.iterrows():
            # Correspondance de domaine
            domain_match = user['domain'] == job['domain']
            domain_weight = 0.5 if domain_match else 0.2
            
            # Correspondance de compétences
            common_skills = set(user['skills']).intersection(set(job['required_skills']))
            skill_match_score = len(common_skills) / len(job['required_skills']) if job['required_skills'] else 0
            
            # Correspondance de localisation
            location_match = user['location'] == job['location']
            location_weight = 0.3 if location_match else 0.1
            
            # Calcul du score de pertinence
            relevance_score = (skill_match_score * 0.6) + (domain_weight) + (location_weight)
            
            if relevance_score > 0.4 or random.random() < 0.1:  # Ajouter un peu de diversité
                relevant_jobs.append((job['id'], relevance_score))
        
        # Trier par pertinence et sélectionner les plus pertinents
        relevant_jobs.sort(key=lambda x: x[1], reverse=True)
        relevant_jobs = relevant_jobs[:min(20, len(relevant_jobs))]
        
        # Générer des interactions pour ces emplois pertinents
        for job_id, relevance in relevant_jobs:
            if random.random() < interaction_rate * relevance * 2:  # Plus de chances pour les emplois pertinents
                interaction = {
                    'user_id': user['id'],
                    'job_id': job_id,
                    'viewed': True,
                    'applied': random.random() < 0.4,
                    'saved': random.random() < 0.3,
                    'date': (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'),
                    'relevance_score': relevance
                }
                interactions.append(interaction)
    
    return pd.DataFrame(interactions)

# 5. Générer les données et les enregistrer
# ----------------------------------------

print("Génération des utilisateurs...")
users_df = generate_users(1000)
print(f"Nombre d'utilisateurs générés: {len(users_df)}")

print("Génération des offres d'emploi...")
jobs_df = generate_jobs(2000)
print(f"Nombre d'offres d'emploi générées: {len(jobs_df)}")

print("Génération des interactions...")
interactions_df = generate_interactions(users_df, jobs_df)
print(f"Nombre d'interactions générées: {len(interactions_df)}")

# Visualisation des données

# Distribution des domaines d'utilisateurs
plt.figure(figsize=(12, 6))
users_domain_counts = users_df['domain'].value_counts()
sns.barplot(x=users_domain_counts.values, y=users_domain_counts.index)
plt.title('Distribution des utilisateurs par domaine')
plt.tight_layout()
plt.show()

# Distribution des compétences les plus courantes
plt.figure(figsize=(12, 6))
skill_counts = {}
for skills_list in users_df['skills']:
    for skill in skills_list:
        skill_counts[skill] = skill_counts.get(skill, 0) + 1

skill_df = pd.DataFrame({
    'skill': list(skill_counts.keys()),
    'count': list(skill_counts.values())
}).sort_values('count', ascending=False)

sns.barplot(x='count', y='skill', data=skill_df.head(15))
plt.title('Compétences les plus courantes parmi les utilisateurs')
plt.tight_layout()
plt.show()

# Distribution des localisations
plt.figure(figsize=(12, 6))
location_counts = users_df['location'].value_counts()
sns.barplot(x=location_counts.values, y=location_counts.index)
plt.title('Distribution des utilisateurs par localisation')
plt.tight_layout()
plt.show()

# Enregistrer les données générées au format CSV
users_df.to_csv('freelance_users_maroc.csv', index=False)
jobs_df.to_csv('freelance_jobs_maroc.csv', index=False)
interactions_df.to_csv('freelance_interactions_maroc.csv', index=False)

print("\nDonnées sauvegardées dans les fichiers CSV:")
print("- freelance_users_maroc.csv")
print("- freelance_jobs_maroc.csv")
print("- freelance_interactions_maroc.csv")

# Exporter également au format JSON pour conserver la structure imbriquée
def dataframe_to_json(df, filename):
    records = df.to_dict(orient='records')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

dataframe_to_json(users_df, 'freelance_users_maroc.json')
dataframe_to_json(jobs_df, 'freelance_jobs_maroc.json')
dataframe_to_json(interactions_df, 'freelance_interactions_maroc.json')

print("\nDonnées également sauvegardées au format JSON:")
print("- freelance_users_maroc.json")
print("- freelance_jobs_maroc.json")
print("- freelance_interactions_maroc.json")

print("\nLe jeu de données a été généré avec succès et est prêt à être utilisé pour l'entraînement de modèles de recommandation.")