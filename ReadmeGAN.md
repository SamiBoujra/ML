GAN - MNIST Dataset
Ce projet implémente un Generative Adversarial Network (GAN) pour générer des images de chiffres manuscrits en utilisant le dataset MNIST. Le code est écrit en PyTorch et présente une architecture de base de GAN composée de deux réseaux neuronaux : un Générateur et un Discriminateur.

À quoi ça sert ?
Le but de ce projet est de démontrer l'utilisation de GANs pour la génération d'images. Plus précisément, le modèle apprend à créer des images de chiffres manuscrits à partir d'un vecteur de bruit aléatoire (espace latent), en utilisant un processus d'apprentissage compétitif où :

Le Générateur tente de créer des images qui ressemblent à de vraies images du dataset MNIST.
Le Discriminateur essaie de différencier les images réelles des images générées par le Générateur.
Fonctionnement du Code
1. Discriminateur (Discriminator)
Le Discriminateur est un réseau neuronal simple qui prend une image en entrée (réelle ou générée) et donne une probabilité que cette image soit réelle. Il utilise une architecture de type Fully Connected (couches linéaires), avec une activation LeakyReLU et une sortie Sigmoid pour produire une probabilité entre 0 et 1.

2. Générateur (Generator)
Le Générateur prend un vecteur de bruit aléatoire (de taille z_dim=64) et le transforme en une image de taille 28x28 pixels (réelle image MNIST). Il utilise également une architecture de type Fully Connected avec une activation LeakyReLU et une fonction de sortie Tanh pour normaliser les images générées dans l'intervalle [-1, 1].

3. Entraînement du Modèle
Le modèle est entraîné pendant 50 époques. À chaque itération, le Discriminateur et le Générateur sont mis à jour :

Le Discriminateur est entraîné pour maximiser la probabilité de classer correctement les images réelles et générées.
Le Générateur est entraîné pour tromper le Discriminateur, c'est-à-dire maximiser la probabilité que les images qu'il génère soient classées comme réelles.
4. Visualisation avec TensorBoard
Le code utilise TensorBoard pour afficher les courbes de perte (loss) pendant l'entraînement, permettant de suivre les progrès du modèle.
