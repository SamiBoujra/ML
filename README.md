# YOLOv1 avec Batch Normalization

Ce projet implémente l'architecture **YOLOv1** (You Only Look Once) pour la détection d'objets en temps réel, avec une légère modification ajoutant des couches de **Batch Normalization**. YOLOv1 est un modèle de détection d'objets qui divise l'image en une grille et prédit des boîtes englobantes et des classes pour chaque cellule.

## Architecture

Le modèle suit l'architecture classique de YOLOv1, mais avec l'ajout de **Batch Normalization** après chaque couche de convolution pour améliorer la convergence et la stabilité de l'entraînement. L'architecture est définie à l'aide de tuples dans la variable `architecture_config`, spécifiant la taille des noyaux de convolution, le nombre de filtres, le stride et le padding.

## Structure du code

Le code est organisé de manière à séparer la définition du modèle, le traitement des données et la fonction de perte :

- **`model.py`** : Contient la définition du modèle YOLOv1 avec les couches de convolution, BatchNorm et activation LeakyReLU.
- **`dataset.py`** : Contient les fonctions de gestion et de prétraitement du dataset. Il permet de charger les images et leurs annotations dans un format adapté au modèle.
- **`loss.py`** : Implémente la fonction de perte spécifique pour YOLO, incluant les erreurs de prédiction pour les boîtes englobantes et les classes.
- **`train.py`** : Un fichier pour entraîner le modèle sur un jeu de données.
- **`utils.py`** : Fonctions utilitaires pour le traitement des images, la gestion des données et le calcul des métriques (par exemple, l'IoU).

## Dataset

Ce projet est conçu pour être utilisé avec un dataset annoté pour la détection d'objets. Le format attendu pour chaque image est une liste de boîtes englobantes associées à des classes d'objets. Chaque boîte englobante est définie par 4 coordonnées : `(x_center, y_center, width, height)` et chaque objet est associé à une classe (identifiée par un numéro).

Les formats d'annotation peuvent être :
- **PASCAL VOC** : Où chaque image a un fichier XML contenant les informations sur les boîtes englobantes et les classes.
- **COCO** : Où les annotations sont fournies sous forme de JSON.

Le fichier **`dataset.py`** s'occupe de charger ces annotations et de les convertir en un format adapté pour l'entraînement.

## Entraînement

Pour entraîner le modèle sur vos propres données, vous pouvez utiliser le fichier `train.py`. Ce fichier configure l'entraînement en chargeant les données via `dataset.py`, en calculant la fonction de perte via `loss.py` et en optimisant le modèle à l'aide de PyTorch.
