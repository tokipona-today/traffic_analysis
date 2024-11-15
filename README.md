# Analyse de Trafic avec OpenCV

Ce projet fait partie d'un cours pratique sur OpenCV et le traitement d'images en Python. Il illustre l'utilisation de techniques de vision par ordinateur pour analyser le trafic routier en temps réel.

## Objectifs Pédagogiques

- Manipulation d'images avec OpenCV
- Soustraction de fond et détection de mouvement
- Segmentation d'image et définition de ROI (Regions of Interest)
- Visualisation de données en temps réel avec Streamlit
- Application pratique des concepts de traitement d'image

## Fonctionnalités

- Détection de véhicules par soustraction de fond
- Analyse par zones d'intérêt configurables
- Calcul du taux d'occupation
- Visualisation du flux de trafic en temps réel
- Statistiques par zone

## Prérequis

- Python 3.8+
- OpenCV
- Streamlit
- NumPy
- PIL

## Installation

```bash
git clone [URL_DU_REPO]
cd analyse-trafic
pip install -r requirements.txt
```

## Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Dans l'interface :
   - Chargez un masque pour définir les zones d'analyse
   - Sélectionnez une source vidéo (webcam, fichier ou URL)
   - Ajustez les paramètres par zone si nécessaire

## Structure du Projet

```
.
├── app.py              # Application principale
├── requirements.txt    # Dépendances
└── .streamlit/        
    └── config.toml    # Configuration Streamlit
```

## Support Pédagogique

Ce code est commenté pour faciliter la compréhension. Les concepts clés abordés :
- Prétraitement d'images
- Détection de mouvement
- Segmentation et masquage
- Interface utilisateur avec Streamlit

## Auteurs
Nikos Priniotakis
