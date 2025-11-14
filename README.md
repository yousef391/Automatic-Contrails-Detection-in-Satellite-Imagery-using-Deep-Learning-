# 🛩️ Détection de Contrails par Deep Learning

> Projet de classification binaire utilisant PyTorch et Transfer Learning pour détecter automatiquement les contrails (traînées de condensation) dans des images satellites.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Résultats](#-résultats)
- [Structure du Projet](#-structure-du-projet)
- [Technologies Utilisées](#-technologies-utilisées)

---

## 🎯 Vue d'ensemble

Ce projet implémente un système de détection automatique de contrails dans des images satellites en utilisant des techniques de Deep Learning avancées. Les contrails (condensation trails) sont des traînées de condensation laissées par les avions et leur détection automatique est importante pour :

- **Études climatologiques** : Impact des contrails sur le climat
- **Surveillance aérienne** : Détection automatique dans les images satellites
- **Recherche aéronautique** : Analyse de l'impact environnemental du trafic aérien

### Objectif

Créer un modèle de classification binaire capable de déterminer si une image satellite contient des contrails ou non :
- **Classe 0** : Pas de contrail dans l'image
- **Classe 1** : Contrails présents dans l'image

---

## ✨ Fonctionnalités

### 🔬 Fonctionnalités Principales

- ✅ **Classification binaire** avec ResNet50 et Transfer Learning
- ✅ **Visualisation Grad-CAM** : Zones d'attention du modèle
- ✅ **Analyse approfondie** : Matrice de confusion, métriques détaillées
- ✅ **Analyse des erreurs** : Faux positifs/négatifs avec visualisations
- ✅ **Distribution des probabilités** : Analyse de confiance du modèle
- ✅ **Pipeline complet** : De la préparation des données à l'évaluation

### 📊 Métriques Évaluées

- **Accuracy** : Précision globale
- **Precision** : Précision des prédictions positives
- **Recall** : Taux de détection des vrais positifs
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **Matrice de confusion** : Analyse détaillée des erreurs

---

## 🏗️ Architecture

### Modèle

- **Architecture de base** : ResNet50 pré-entraîné sur ImageNet
- **Transfer Learning** : Adaptation de la dernière couche pour classification binaire
- **Fine-tuning** : Ajustement des poids sur le dataset de contrails

### Pipeline de Données

```
Images PNG → Preprocessing → Augmentation → Normalisation → ResNet50 → Classification
```

### Techniques Utilisées

- **Data Augmentation** : Flip horizontal, rotation, ajustement couleur
- **Normalisation** : Statistiques ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Optimization** : Adam optimizer avec learning rate 0.001
- **Loss Function** : CrossEntropyLoss

---

## 🚀 Installation

### Prérequis

- Python 3.8+
- PyTorch 2.0+
- CUDA (optionnel, pour GPU)

### Installation des Dépendances

```bash
pip install torch torchvision
pip install numpy pillow matplotlib
pip install scikit-learn seaborn
```

### Structure du Dataset

Organisez vos données comme suit :

```
SingleFrame_PNG/
├── train/
│   ├── images/          # Images PNG/JPG
│   └── ground_truth/     # Labels .npy
├── validation/
│   ├── images/
│   └── ground_truth/
└── test/
    ├── images/
    └── ground_truth/
```

---

## 💻 Utilisation

### 1. Configuration

Ouvrez le notebook `atmo_class.ipynb` et ajustez le chemin du dataset :

```python
DATA_DIR = 'SingleFrame_PNG'  # Votre chemin
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
```

### 2. Exécution

Exécutez les cellules du notebook dans l'ordre :

1. **Imports et configuration**
2. **Dataset et DataLoader**
3. **Préparation des données**
4. **Modèle ResNet50**
5. **Entraînement**
6. **Évaluation**
7. **Grad-CAM et visualisations**
8. **Analyse des erreurs**

### 3. Charger un Modèle Sauvegardé

```python
checkpoint = torch.load('contrails_classifier_resnet50.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Faire une prédiction
model.eval()
with torch.no_grad():
    output = model(image_tensor)
    prediction = torch.argmax(output, dim=1)
    probability = torch.softmax(output, dim=1)
```

---

## 📈 Résultats

### Métriques Typiques

Sur un dataset équilibré, le modèle atteint généralement :

- **Accuracy** : > 85%
- **F1-Score** : > 0.80
- **Precision** : > 0.82
- **Recall** : > 0.78

*Note : Les résultats varient selon la qualité et la taille du dataset.*

### Visualisations

Le notebook génère automatiquement :

1. **Courbes d'apprentissage** : Évolution de la loss et accuracy
2. **Grad-CAM heatmaps** : Zones d'attention du modèle
3. **Matrice de confusion** : Analyse des erreurs
4. **Distribution des probabilités** : Confiance du modèle
5. **Exemples d'erreurs** : Faux positifs/négatifs avec explications

---

## 📁 Structure du Projet

```
atmo/
├── atmo_class.ipynb          # Notebook principal
├── README.md                 # Ce fichier
├── PROJECT_EXPLANATION.md    # Explication détaillée du projet
├── INTERNSHIP_EVALUATION.md # Évaluation pour stage
└── contrails_classifier_resnet50.pth  # Modèle sauvegardé (généré)
```

---

## 🛠️ Technologies Utilisées

- **PyTorch** : Framework de Deep Learning
- **Torchvision** : Modèles pré-entraînés et transformations
- **NumPy** : Calculs numériques
- **PIL/Pillow** : Traitement d'images
- **Matplotlib** : Visualisations
- **Seaborn** : Visualisations statistiques
- **Scikit-learn** : Métriques d'évaluation

---

## 🎓 Points Techniques Clés

### Transfer Learning

Le modèle utilise ResNet50 pré-entraîné sur ImageNet, ce qui permet :
- **Apprentissage rapide** : Moins d'époques nécessaires
- **Meilleures performances** : Avec moins de données
- **Généralisation** : Patterns visuels déjà appris

### Grad-CAM

Implémentation de Gradient-weighted Class Activation Mapping pour :
- **Interprétabilité** : Comprendre où le modèle regarde
- **Validation** : Vérifier que le modèle se concentre sur les bonnes régions
- **Debugging** : Identifier les biais potentiels

### Analyse des Erreurs

Analyse approfondie pour :
- **Comprendre les limites** du modèle
- **Identifier les cas difficiles**
- **Guider les améliorations futures**

---

## 🔮 Améliorations Futures

- [ ] **Segmentation U-Net** : Localisation précise des contrails
- [ ] **Ensemble de modèles** : Combinaison de plusieurs architectures
- [ ] **Fine-tuning avancé** : Learning rate scheduling, early stopping
- [ ] **Augmentation avancée** : Mixup, CutMix
- [ ] **Architectures alternatives** : EfficientNet, Vision Transformer

---

## 📝 Auteur

Projet développé dans le cadre d'une candidature pour un stage de recherche.

---

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

## 🙏 Remerciements

- **ONERA** pour le contexte d'application
- **PyTorch Team** pour le framework
- **ImageNet** pour les poids pré-entraînés

---

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue.

---

**⭐ Si ce projet vous a aidé, n'hésitez pas à le star sur GitHub !**

