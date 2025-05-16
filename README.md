# Projet de reconnaissance d'expressions faciales — Installation de l'environnement

Ce projet utilise **PyTorch** pour entraîner un modèle de Deep Learning (ex : DenseNet121) à reconnaître les expressions faciales.  
Ce guide vous accompagne pour installer l'environnement Python, y compris l'accélération GPU via **CUDA** si vous avez une carte NVIDIA.

---

##  Prérequis

- Python **3.8 à 3.12** recommandé
- pip
- Une carte graphique NVIDIA (optionnelle, mais recommandée pour l'accélération CUDA)
- Linux, Windows ou WSL

---

##  1. Créer un environnement virtuel (fortement recommandé)

### Linux / Mac :

```bash```
python3 -m venv env
source env/bin/activate

Windows
python -m venv env
.\env\Scripts\activate

# 2. Mettre pip à jour
```bash```
Copier
Modifier

# 3. Installation de PyTorch (CPU ou GPU)
  ➤ A. Détecter si vous avez une carte NVIDIA
    ```bash```
      nvidia-smi

✅ Si une table s'affiche avec le nom de la carte graphique : vous avez une carte NVIDIA compatible CUDA.

❌ Sinon, passez à l'installation CPU (voir plus bas).


  ➤ B. Trouver votre version de CUDA (si installée)
    nvcc --version
    ou
    cat /usr/local/cuda/version.txt
  Cela vous donne une version comme : release 11.8 → à noter.

  ➤ C. Installer PyTorch avec pip
    Exemple CUDA 12.1 :
      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Installer les dépendances du projet
  bash : pip install -r requirements.txt

# 6. Entraînement et exécution
  Vous pouvez maintenant :
    Entraîner votre modèle avec train.py
    Tester la détection avec detection_camera_densenet_ft.py
    Connecter au robot Furhat (voir la section dédiée si besoin)



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


Fait avec ❤️ par [Guidon Antoine]
