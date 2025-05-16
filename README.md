#  Reconnaissance d’Émotions Faciales avec DenseNet121

Ce projet implémente un système de reconnaissance d’expressions faciales (angry, happy, neutral, sad) basé sur un modèle **DenseNet121** fine-tuné, destiné à être utilisé notamment avec un robot Furhat.

---

##  Objectif du projet

- Charger une image faciale
- Prédire son expression
- Intégrer les résultats dans un système interactif (robot, interface, etc.)

---

##  Modèle utilisé : DenseNet121

- **Base** : pré-entraîné sur ImageNet
- **Fine-tuning** : uniquement à partir du bloc `denseblock3`
- **Sortie personnalisée** : 4 classes (`angry`, `happy`, `neutral`, `sad`)

#  INSTALLATION DE L'ENVIRONNEMENT

Ce projet repose sur **PyTorch**, **Torchvision**, et d’autres librairies Python.  
Tu peux l’installer rapidement de deux façons : avec `uv` (recommandé) ou manuellement avec `venv` + `pip`.

---

##  OPTION 1 – Installation Rapide avec `uv` (recommandé ⚡)

[`uv`](https://github.com/astral-sh/uv) est un gestionnaire moderne de paquets Python ultra-rapide, compatible `pip`.

###  Étape 1 : Installer `uv`

####  Linux / macOS :
```bash```
curl -Ls https://astral.sh/uv/install.sh | sh

 macOS avec Homebrew :
 ```bash``` brew install astral-sh/uv/uv
### Étape 2 : Créer un environnement virtuel et l’activer
```bash```
  uv venv
  source .venv/bin/activate        # Linux / macOS
  .venv\Scripts\activate           # Windows
### Étape 3 : Installer les dépendances CPU
```bash```
  uv pip install -r requirements.txt
### Étape 4 (optionnelle) : Installer PyTorch avec support CUDA
  Si tu as une carte graphique NVIDIA, installe la version adaptée :

CUDA 12.1
      ```bash```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CUDA 11.8
```bash```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


## OPTION 2 – Installation Manuelle avec python -m venv + pip
### Étape 1 : Créer un environnement virtuel
```bash```
python -m venv env
source env/bin/activate         # Linux / macOS
env\Scripts\activate            # Windows
### Étape 2 : Mettre pip à jour
```bash``` pip install --upgrade pip
### Étape 3 : Installer les dépendances CPU
```bash```
  pip install -r requirements.txt
  
### Étape 4 (optionnelle) : Installer PyTorch avec CUDA
Visite : https://pytorch.org/get-started/locally/
Ou installe directement :

  CUDA 12.1
```bash```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CUDA 11.8











# Facial Expression Recognition – 4 Classes

  Objectif : 
    Ce projet a pour but de développer un modèle de reconnaissance d'expressions faciales capable de classer les émotions humaines en 4 catégories :
      -Joie (Happy)
      -Tristesse (Sad)
      -Colère (Angry)
      -Neutre (Neutral)
  Il est conçu pour des applications en temps réel via webcam, en utilisant des techniques de vision par ordinateur et d'apprentissage profond


Description du projet :
  Le projet repose sur un pipeline complet de traitement d'images faciales :
    Détection de visage : Utilisation de Haar Cascades pour localiser les visages dans les images.
    Prétraitement : Redimensionnement et normalisation des visages détectés.
    Classification : Modèle de réseau de neurones convolutif (CNN) entraîné pour prédire l'expression faciale parmi les 4 classes.
    Le modèle est entraîné sur un sous-ensemble du dataset FER2013, adapté pour les 4 classes ciblées.

Structure du dépôt:
| Élément                          | Type        | Description                                                                                                                                              |
| -------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DenseNet121/`                   |  Dossier  | Contient l’architecture du modèle DenseNet121, utilisée comme base pour le transfert learning.                                                             |
| `FER_2013/`                      |  Dossier  | Version prétraitée du dataset FER2013, filtrée pour ne conserver que 4 classes : `angry`, `happy`, `neutral`, `sad`. Utilisée pour l'entraînement initial. |
| `captured_faces/`                |  Dossier  | Contient des visages capturés en direct (par webcam ou robot) utilisés pour l’évaluation.                                                                  |
| `captured_faces_finetunning/`    |  Dossier  | Données personnelles capturées pour le fine-tuning du modèle afin de mieux s’adapter à des visages spécifiques ou en conditions réelles.                   |
| `finetuned_model/`               |  Dossier  | Modèle(s) entraîné(s) et affiné(s) (`.pth`) prêt(s) à être utilisé(s) pour la prédiction.                                                                  |
| `savedir/`                       |  Dossier  | Dossier temporaire où sont stockés les résultats des prédictions : images annotées, logs, etc.                                                             |
| `DenseNet_4classe.ipynb`         |  Notebook | Implémentation complète du pipeline : chargement du dataset, entraînement de DenseNet121, évaluation. C’est le notebook principal du projet.               |
| `finetuning_modele_donnee.ipynb` |  Notebook | Permet de recharger un modèle existant et de le **fine-tuner** sur de nouvelles données personnalisées (capturées dans `captured_faces_finetunning/`).     |
| `fine_tuned_model_data_final.pth`|  Archive  | Archive contenant un modèle déjà fine-tuné, directement utilisable pour tester les prédictions sans passer par un nouvel entraînement.                     |
| `demo.py`                        |  Code     | Code de démonstration de détection d'émotion avec le model définit.                                                                                        |



