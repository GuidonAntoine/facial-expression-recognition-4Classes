Facial Expression Recognition – 4 Classes

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
