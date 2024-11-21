# Projet ETL - Classement des Banques Mondiales

## Description
Script Python pour extraire, transformer et charger des données sur les plus grandes banques mondiales depuis Wikipedia, avec conversion de devises.

## Fonctionnalités
- Extraction des données bancaires depuis web.archive.org
- Transformation avec conversion de devises (USD, GBP, EUR, INR)
- Sauvegarde en CSV et base de données SQLite
- Journalisation des étapes du processus

## Prérequis
- Python 3.x
- Bibliothèques : 
  - pandas
  - numpy
  - requests
  - beautifulsoup4
  - sqlite3

## Configuration
- Chemins de fichiers à configurer :
  - `log_file`: Fichier de logs
  - `csv_file`: Export CSV
  - `exchange_rates`: Taux de change
- Base de données : `Banks.db`
- Table : `Largest_banks`

## Fonctions Principales
- `extract()`: Récupère les données des banques depuis Wikipedia
- `transform()`: Convertit les valeurs de marché en différentes devises
- `load_to_csv()`: Sauvegarde en fichier CSV
- `load_to_db()`: Charge dans une base SQLite
- `run_query()`: Exécute des requêtes SQL

## Queries Exemples
- Sélection de toutes les données
- Moyenne de capitalisation en GBP
- Sélection des 5 premières banques

## Utilisation
```bash
python bank_projects.py
```

## Logs
Suivi détaillé dans `code_log.txt`
