# IA D�fenseur Adaptatif : Strat�gies Mixtes pour la S�curit� Face aux Attaques �volutives

## Description
Ce projet traite un probl�me d'intelligence artificielle o� un D�fenseur apprend � adapter sa strat�gie face � un Hacker dans un jeu r�p�titif.

## Installation

### Installation des d�pendances
`powershell
pip install -r requirements.txt
`

## Utilisation

### Option 1: Interface Graphique (Recommand�) 
`powershell
python run_gui.py
`

Une interface graphique intuitive qui permet de:
- Configurer les param�tres de simulation avec curseurs
- Voir la progression en temps r�el
- Visualiser les r�sultats directement dans l'interface

### Option 2: Ligne de commande
`powershell
python main.py
`

Ex�cute une simulation et g�n�re 9 images PNG avec les graphiques.

### Option 3: Tests
`powershell
python test_modules.py
`

V�rifie que tous les modules fonctionnent.

## Fichiers principaux
- gui.py: Interface graphique pygame
- main.py: Script de simulation en ligne de commande
- game_theory_defender.py: M�canique du jeu
- defender_ai.py: IA du d�fenseur (Q-Learning)
- hacker_ai.py: IA du hacker (Adaptif/R�actif)
- utils.py: Utilitaires et statistiques
