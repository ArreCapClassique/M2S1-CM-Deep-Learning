# Deep Learning – Questions de cours (M2 MIAGE IDA)

---

## 1. Réseaux de neurones artificiels (ANN / Perceptron)

### Q1. Définissez un perceptron et expliquez son fonctionnement.
**Réponse :**  
Un perceptron est le modèle fondamental d’un neurone artificiel. Il réalise une **combinaison linéaire** des entrées pondérées, à laquelle on ajoute un biais, puis applique une **fonction d’activation**.  
Mathématiquement :  
\[
y = f\left(\sum_i w_i x_i + b\right)
\]  
Le perceptron permet de résoudre des problèmes de **classification linéairement séparables**.

---

### Q2. Quel est le rôle du biais dans un réseau de neurones ?
**Réponse :**  
Le biais permet de **déplacer la frontière de décision** indépendamment des entrées. Sans biais, le modèle serait contraint de passer par l’origine, ce qui réduit fortement sa capacité de représentation.  
Il joue un rôle analogue à l’ordonnée à l’origine dans une régression linéaire.

---

### Q3. Qu’est-ce qu’un perceptron multicouche (MLP) ?
**Réponse :**  
Un perceptron multicouche est un réseau de neurones composé :
- d’une couche d’entrée,
- d’une ou plusieurs couches cachées,
- d’une couche de sortie,  
avec des connexions **entièrement connectées** entre couches successives.  
Dès que le réseau contient **au moins deux couches cachées**, on parle généralement de **deep learning**.

---

## 2. Fonctions d’activation

### Q4. Qu’est-ce qu’une fonction d’activation ?
**Réponse :**  
Une fonction d’activation introduit de la **non-linéarité** dans le réseau, permettant de modéliser des relations complexes entre les données. Sans elle, un réseau profond serait équivalent à un modèle linéaire.

---

### Q5. Comparez la fonction sigmoïde et la fonction ReLU.
**Réponse :**
- **Sigmoïde** : sortie dans ]0,1[, utilisée historiquement ; souffre du problème de **gradient qui disparaît**.
- **ReLU** : \( f(x)=\max(0,x) \) ; plus simple, convergence plus rapide, largement utilisée dans les réseaux profonds modernes.

---

## 3. Apprentissage et optimisation

### Q6. Décrivez les étapes de l’entraînement d’un réseau de neurones.
**Réponse :**
1. Initialisation des poids  
2. Propagation avant (forward pass)  
3. Calcul de la fonction de perte  
4. Rétropropagation du gradient  
5. Mise à jour des poids (descente de gradient)

---

### Q7. Quel est le rôle de la fonction de perte ?
**Réponse :**  
La fonction de perte mesure l’écart entre la prédiction du modèle et la vérité terrain. Elle guide l’apprentissage via le calcul du gradient.  
Exemples :
- **Cross-entropy** : classification
- **MSE** : régression

---

## 4. Overfitting, underfitting et régularisation

### Q8. Différence entre overfitting et underfitting.
**Réponse :**
- **Underfitting** : modèle trop simple, incapable de capturer la structure des données.
- **Overfitting** : modèle trop complexe, excellent sur les données d’entraînement mais mauvais en généralisation.

---

### Q9. Citez des techniques de régularisation.
**Réponse :**
- Dropout  
- L1 / L2 (weight decay)  
- Early stopping  
- Data augmentation

---

## 5. Convolution et CNN

### Q10. Expliquez le principe de la convolution dans les CNN.
**Réponse :**  
La convolution applique un **filtre local** sur l’image afin d’extraire des motifs spatiaux (bords, textures, formes).  
Elle permet :
- la **localité**,
- le **partage des poids**,
- une forte réduction du nombre de paramètres.

:contentReference[oaicite:3]{index=3}

---

### Q11. Quel est l’intérêt du pooling ?
**Réponse :**  
Le pooling (max ou average) permet de :
- réduire la dimension spatiale,
- rendre le modèle plus robuste aux translations,
- diminuer le coût de calcul.

---

### Q12. Rôle des couches dans un CNN.
**Réponse :**
- **Convolution** : extraction de features
- **Pooling** : réduction et invariance
- **Fully connected** : décision finale

---

## 6. Architectures avancées

### Q13. Pourquoi utiliser des connexions résiduelles (ResNet) ?
**Réponse :**  
Les connexions résiduelles permettent de :
- faciliter la propagation du gradient,
- éviter la dégradation des performances quand le réseau devient profond,
- conserver des features de bas niveau.

---

### Q14. Expliquez le principe d’un auto-encodeur.
**Réponse :**  
Un auto-encodeur apprend à **reconstruire son entrée** via :
- un encodeur (compression),
- un espace latent,
- un décodeur (reconstruction).  
Applications : débruitage, réduction de dimension, super-résolution.

---

### Q15. Décrivez l’architecture U-Net.
**Réponse :**  
U-Net est une architecture en forme de U utilisée pour la **segmentation sémantique**.  
Elle combine :
- un chemin encodeur (convolutions + pooling),
- un chemin décodeur,
- des **skip connections** reliant niveaux symétriques.

---

## 7. Séries temporelles et RNN

### Q16. Pourquoi les RNN sont adaptés aux séries temporelles ?
**Réponse :**  
Ils prennent en compte la **dépendance temporelle** grâce à un état caché récurrent partagé entre les pas de temps.

---

### Q17. Différence entre RNN et LSTM.
**Réponse :**
- RNN classiques : souffrent du gradient qui disparaît.
- LSTM : introduisent des **portes (input, forget, output)** et une mémoire longue durée.

:contentReference[oaicite:4]{index=4}

---

### Q18. Pourquoi préférer un LSTM à un CNN pour une série temporelle ?
**Réponse :**  
Le LSTM capture explicitement l’ordre et les dépendances longues, contrairement au CNN qui traite surtout des motifs locaux.

---

## 8. GAN et génération

### Q19. Expliquez le principe d’un GAN.
**Réponse :**  
Un GAN repose sur deux réseaux adverses :
- **Générateur** : crée des données synthétiques,
- **Discriminateur** : distingue vrai/faux.  
L’entraînement est un jeu à somme nulle visant à tromper le discriminateur.

---

### Q20. Comment évaluer un modèle génératif ?
**Réponse :**  
L’évaluation est difficile et souvent qualitative :
- inspection visuelle,
- diversité des échantillons,
- stabilité de l’apprentissage.

---

## 9. Transformers et LLM

### Q21. Quel est le principe d’un Transformer ?
**Réponse :**  
Les Transformers reposent sur un **mécanisme d’attention**, permettant de pondérer l’importance des éléments d’une séquence sans récurrence.  
Ils sont plus parallélisables que les RNN.

---

### Q22. En quoi l’attention est-elle centrale ?
**Réponse :**  
L’attention permet au modèle de se concentrer sur les tokens pertinents du contexte pour prédire la suite, améliorant cohérence et performance.

---

## 10. Transfer learning

### Q23. Qu’est-ce que le transfer learning ?
**Réponse :**  
Le transfer learning consiste à réutiliser un modèle pré-entraîné sur un large dataset et à l’adapter à une nouvelle tâche via le **fine-tuning**.

---

### Q24. Pourquoi le transfer learning est-il efficace ?
**Réponse :**
- Moins de données nécessaires,
- Temps de calcul réduit,
- Réutilisation de features génériques (bords, textures, structures).

---

## 11. Interprétabilité

### Q25. Qu’est-ce que Grad-CAM ?
**Réponse :**  
Grad-CAM produit une **carte de chaleur** indiquant les régions de l’image ayant le plus contribué à la décision du modèle.

:contentReference[oaicite:5]{index=5}

---

### Q26. Pourquoi l’explicabilité est-elle essentielle ?
**Réponse :**
- validation scientifique,
- détection de biais,
- confiance dans le modèle,
- conformité réglementaire.

---

## 12. Choix d’architecture

### Q27. Comment choisir une architecture selon le type de données ?
**Réponse :**
- Données tabulaires → MLP  
- Images → CNN / ResNet  
- Séries temporelles → LSTM / GRU  
- Texte → Transformer  

---


# 13. Données, prétraitement et représentation

## Q28. Pourquoi le prétraitement des données est-il crucial en deep learning ?
**Réponse :**  
Le deep learning est extrêmement sensible à la qualité des données. Un mauvais prétraitement peut empêcher la convergence ou induire des biais forts.  
Le prétraitement vise à :
- homogénéiser les échelles (normalisation, standardisation),
- réduire le bruit,
- gérer les valeurs manquantes,
- rendre les données compatibles avec l’architecture choisie.

Dans le cas des images, on redimensionne et normalise les pixels ; pour le texte, on tokenize et encode ; pour les séries temporelles, on aligne et segmente les séquences.

---

## Q29. Quelle est la différence entre normalisation et standardisation ?
**Réponse :**
- **Normalisation** : mise à l’échelle dans un intervalle fixe (ex. [0,1]).
- **Standardisation** : recentrage (moyenne 0) et réduction (variance 1).

La standardisation est souvent préférable pour les réseaux utilisant ReLU, tandis que la normalisation est fréquente pour les images.

---

## Q30. Pourquoi l’encodage des données catégorielles est-il nécessaire ?
**Réponse :**  
Les réseaux de neurones ne traitent que des valeurs numériques.  
Les variables catégorielles doivent être transformées :
- **One-hot encoding** : simple mais coûteux en dimension,
- **Embeddings** : représentation dense apprise, essentielle en NLP.

---

# 14. Initialisation des poids et stabilité numérique

## Q31. Pourquoi l’initialisation des poids est-elle importante ?
**Réponse :**  
Une mauvaise initialisation peut provoquer :
- explosion du gradient,
- disparition du gradient,
- convergence lente ou échec de l’apprentissage.

Les initialisations modernes (Xavier, He) tiennent compte du nombre de neurones entrants et sortants afin de stabiliser la variance des activations.

---

## Q32. Différence entre initialisation Xavier et He.
**Réponse :**
- **Xavier (Glorot)** : adaptée aux activations sigmoïdes ou tanh.
- **He** : optimisée pour ReLU et ses variantes.

---

# 15. Optimisation et algorithmes de descente

## Q33. Qu’est-ce que la descente de gradient stochastique (SGD) ?
**Réponse :**  
La SGD met à jour les poids à partir de sous-ensembles (mini-batchs) de données, ce qui :
- réduit le coût mémoire,
- introduit du bruit bénéfique pour l’optimisation,
- améliore la généralisation.

---

## Q34. Pourquoi utiliser Adam plutôt que SGD ?
**Réponse :**  
Adam combine :
- momentum,
- adaptation automatique du taux d’apprentissage.  

Il converge plus rapidement et nécessite moins de réglages manuels, ce qui le rend très populaire en pratique.

---

## Q35. Quel est le rôle du learning rate ?
**Réponse :**  
Le learning rate contrôle l’amplitude des mises à jour :
- trop grand → divergence,
- trop petit → convergence très lente.  

Des stratégies comme le **learning rate scheduling** ou le **warm-up** sont souvent utilisées.

---

# 16. Batch size et dynamique d’apprentissage

## Q36. Quel est l’impact de la taille du batch ?
**Réponse :**
- Petit batch : apprentissage bruité, meilleure généralisation.
- Grand batch : apprentissage stable, mais risque de sur-apprentissage et besoin mémoire accru.

Il n’existe pas de valeur universelle optimale.

---

# 17. CNN – aspects avancés

## Q37. Pourquoi les CNN exploitent-ils la structure spatiale des images ?
**Réponse :**  
Les images présentent des corrélations locales.  
La convolution exploite cette propriété en apprenant des filtres locaux réutilisés sur toute l’image, ce qui améliore l’efficacité et la robustesse.

---

## Q38. Qu’est-ce que l’invariance par translation ?
**Réponse :**  
Un CNN est capable de détecter un motif indépendamment de sa position exacte dans l’image, grâce au partage des poids et au pooling.

---

## Q39. Différence entre padding "valid" et "same".
**Réponse :**
- **Valid** : pas de padding, réduction de la taille spatiale.
- **Same** : padding ajouté pour conserver la taille de l’entrée.

---

# 18. Segmentation et détection d’objets

## Q40. Différence entre classification, détection et segmentation.
**Réponse :**
- **Classification** : une étiquette pour l’image entière.
- **Détection** : localisation + classe (boîtes englobantes).
- **Segmentation** : classification au niveau du pixel.

---

## Q41. Pourquoi la segmentation est-elle plus difficile que la détection ?
**Réponse :**  
Elle nécessite une prédiction dense pixel par pixel, donc :
- plus de sorties,
- plus de données annotées,
- plus de complexité computationnelle.

---

# 19. Séries temporelles – approfondissement

## Q42. Pourquoi les CNN peuvent parfois être utilisés sur des séries temporelles ?
**Réponse :**  
Les CNN peuvent extraire des motifs locaux (tendances courtes, saisonnalité), mais ils ne modélisent pas naturellement les dépendances longues, contrairement aux LSTM.

---

## Q43. Rôle du dropout dans les LSTM.
**Réponse :**  
Le dropout réduit l’overfitting en désactivant aléatoirement certaines connexions, mais il doit être utilisé avec précaution pour ne pas perturber la mémoire temporelle.

---

# 20. GAN – difficultés et limites

## Q44. Qu’est-ce que le mode collapse ?
**Réponse :**  
Le générateur produit toujours des sorties très similaires, perdant la diversité du dataset.  
C’est l’un des problèmes majeurs des GAN.

---

## Q45. Pourquoi l’entraînement des GAN est-il instable ?
**Réponse :**  
Il s’agit d’un jeu adversarial non convexe :
- équilibre difficile entre générateur et discriminateur,
- gradients instables,
- oscillations fréquentes.

---

# 21. Transformers et NLP

## Q46. Pourquoi les Transformers ont remplacé les RNN en NLP ?
**Réponse :**
- parallélisation efficace,
- meilleure modélisation des dépendances longues,
- entraînement plus rapide à grande échelle.

---

## Q47. Qu’est-ce qu’un embedding ?
**Réponse :**  
Un embedding est une représentation vectorielle dense d’un token (mot, sous-mot, caractère) capturant des relations sémantiques et syntaxiques.

---

## Q48. Qu’est-ce que le problème de fenêtre de contexte ?
**Réponse :**  
Les Transformers ont une longueur maximale de contexte ; au-delà, l’information est perdue ou approximée, ce qui limite la mémoire à long terme.

---

# 22. Évaluation et métriques

## Q49. Pourquoi l’accuracy est-elle insuffisante dans certains cas ?
**Réponse :**  
En cas de classes déséquilibrées, l’accuracy peut être trompeuse.  
Des métriques comme précision, rappel, F1-score ou AUC sont alors préférables.

---

## Q50. Comment évaluer un modèle de prévision temporelle ?
**Réponse :**
- RMSE, MAE,
- analyse des résidus,
- capacité à suivre la tendance et la saisonnalité.

---

# 23. Pipeline industriel de deep learning

## Q51. Pourquoi le modèle n’est-il qu’une partie du pipeline ?
**Réponse :**  
La performance finale dépend aussi :
- des données,
- du prétraitement,
- de l’évaluation,
- du déploiement,
- du monitoring en production.

---

## Q52. Pourquoi le deep learning est coûteux en calcul ?
**Réponse :**
- grand nombre de paramètres,
- itérations multiples,
- besoin de GPU/TPU,
- volumes massifs de données.

---

# 24. Limites et esprit critique

## Q53. Pourquoi un modèle deep learning peut-il apprendre de mauvais biais ?
**Réponse :**  
Il apprend ce qui est statistiquement corrélé, pas ce qui est causal.  
Des biais dans les données conduisent à des décisions erronées.

---

## Q54. Pourquoi l’explicabilité est-elle limitée en deep learning ?
**Réponse :**  
Les réseaux effectuent une succession massive de transformations non linéaires, rendant l’interprétation exacte humainement impossible à ce jour.

---

# 25. Questions méta (souvent en réflexion)

## Q55. Pourquoi ne pas toujours utiliser le deep learning ?
**Réponse :**
- besoin de grandes quantités de données,
- coût élevé,
- manque d’interprétabilité,
- solutions classiques parfois suffisantes.

---

## Q56. Comment choisir entre ML classique et deep learning ?
**Réponse :**
- peu de données → ML classique,
- données complexes/non structurées → deep learning,
- contraintes explicabilité → modèles simples.

---

# 26. Représentation, espace latent et features

## Q57. Qu’appelle-t-on un espace latent en deep learning ?
**Réponse :**  
L’espace latent est une représentation intermédiaire compacte des données, apprise automatiquement par le réseau.  
Il encode les **features pertinentes** nécessaires à la tâche, tout en éliminant l’information redondante ou non utile.

Dans un CNN, l’espace latent correspond aux cartes de caractéristiques profondes ;  
dans un auto-encodeur, il s’agit du goulot d’étranglement (bottleneck).

---

## Q58. Pourquoi dit-on que le deep learning apprend des features automatiquement ?
**Réponse :**  
Contrairement au machine learning classique, où les features sont conçues manuellement, le deep learning apprend une hiérarchie de représentations :
- features simples (bords, motifs),
- features intermédiaires (formes),
- features complexes (objets, concepts).

Cette capacité est la clé de son efficacité sur les données non structurées.

---

## Q59. Quelle est la différence entre feature engineering et representation learning ?
**Réponse :**
- **Feature engineering** : conception manuelle de variables.
- **Representation learning** : apprentissage automatique des représentations par le modèle.

Le deep learning repose presque exclusivement sur le representation learning.

---

# 27. Profondeur, largeur et capacité des réseaux

## Q60. Quelle est la différence entre un réseau profond et un réseau large ?
**Réponse :**
- Réseau **profond** : beaucoup de couches, hiérarchie de représentations.
- Réseau **large** : peu de couches mais beaucoup de neurones par couche.

Les réseaux profonds sont généralement plus efficaces pour représenter des fonctions complexes.

---

## Q61. Pourquoi augmenter la profondeur plutôt que la largeur ?
**Réponse :**  
La profondeur permet de factoriser les représentations et d’exprimer des fonctions complexes avec moins de paramètres qu’un réseau uniquement large.

---

## Q62. Existe-t-il un risque à augmenter excessivement la profondeur ?
**Réponse :**  
Oui :
- disparition/explosion du gradient,
- surcoût computationnel,
- sur-apprentissage,
- difficulté d’optimisation.

Les architectures modernes (ResNet, skip connections) visent à limiter ces problèmes.

---

# 28. Fonctions de perte – approfondissement

## Q63. Pourquoi ne pas utiliser la MSE pour la classification ?
**Réponse :**  
La MSE n’est pas adaptée aux probabilités de classes car :
- elle pénalise mal les erreurs de classification,
- elle conduit à une convergence plus lente.

La cross-entropy est mieux alignée avec la maximisation de la vraisemblance.

---

## Q64. Différence entre binary cross-entropy et categorical cross-entropy.
**Réponse :**
- **Binary cross-entropy** : classification binaire.
- **Categorical cross-entropy** : classification multi-classes (one-hot).
- **Sparse categorical cross-entropy** : labels entiers, plus efficace en mémoire.

---

# 29. Régularisation – approfondissement

## Q65. Comment le dropout agit-il comme régularisateur ?
**Réponse :**  
Le dropout empêche la co-adaptation des neurones en en désactivant aléatoirement une fraction à chaque itération.  
Il force le réseau à apprendre des représentations plus robustes et redondantes.

---

## Q66. Pourquoi le dropout est-il désactivé en phase de test ?
**Réponse :**  
Lors de l’inférence, on souhaite exploiter l’intégralité du réseau et obtenir une prédiction déterministe.  
Les poids sont implicitement rééchelonnés.

---

# 30. Données déséquilibrées

## Q67. Pourquoi les datasets déséquilibrés posent-ils problème ?
**Réponse :**  
Le modèle peut apprendre à prédire systématiquement la classe majoritaire tout en affichant une accuracy élevée mais trompeuse.

---

## Q68. Comment traiter un dataset déséquilibré ?
**Réponse :**
- pondération des classes,
- sur/sous-échantillonnage,
- métriques adaptées,
- focal loss.

---

# 31. Data augmentation

## Q69. Qu’est-ce que la data augmentation ?
**Réponse :**  
La data augmentation consiste à générer artificiellement de nouvelles données à partir des données existantes (rotations, translations, bruit, etc.) afin d’améliorer la généralisation.

---

## Q70. Pourquoi la data augmentation est-elle efficace en vision par ordinateur ?
**Réponse :**  
Elle introduit des invariances réalistes (position, orientation, éclairage) sans nécessiter de nouvelles annotations.

---

# 32. Séries temporelles – préparation des données

## Q71. Pourquoi découper une série temporelle en fenêtres ?
**Réponse :**  
Les réseaux attendent des entrées de taille fixe.  
Le découpage en fenêtres permet d’apprendre des dépendances locales et globales sur des segments temporels.

---

## Q72. Différence entre prédiction one-step et multi-step.
**Réponse :**
- **One-step** : prédiction du pas suivant.
- **Multi-step** : prédiction de plusieurs pas futurs, plus difficile et instable.

---

# 33. LSTM – approfondissement

## Q73. Rôle de la cellule mémoire dans un LSTM.
**Réponse :**  
La cellule mémoire permet de conserver l’information pertinente sur de longues séquences, limitant la perte d’information temporelle.

---

## Q74. Pourquoi les portes sont-elles nécessaires ?
**Réponse :**  
Les portes contrôlent le flux d’information (écriture, conservation, lecture), ce qui stabilise l’apprentissage.

---

# 34. Comparaison GRU / LSTM

## Q75. Différence entre GRU et LSTM.
**Réponse :**
- GRU : architecture plus simple, moins de paramètres.
- LSTM : plus expressif, mais plus coûteux.

Le choix dépend du compromis performance/coût.

---

# 35. Détection d’anomalies

## Q76. Comment utiliser un auto-encodeur pour la détection d’anomalies ?
**Réponse :**  
L’auto-encodeur apprend à reconstruire les données normales.  
Une erreur de reconstruction élevée indique une anomalie.

---

## Q77. Pourquoi cette approche est-elle non supervisée ?
**Réponse :**  
Elle ne nécessite pas d’exemples annotés d’anomalies, souvent rares ou inconnus.

---

# 36. Déploiement et production

## Q78. Quelles sont les difficultés du déploiement d’un modèle deep learning ?
**Réponse :**
- latence,
- consommation mémoire,
- dérive des données,
- maintenance et mise à jour.

---

## Q79. Qu’est-ce que la dérive des données (data drift) ?
**Réponse :**  
La distribution des données change au cours du temps, rendant le modèle progressivement moins performant.

---

# 37. Éthique et responsabilité

## Q80. Quels sont les risques éthiques du deep learning ?
**Réponse :**
- biais discriminatoires,
- décisions opaques,
- dépendance excessive à l’automatisation.

---

## Q81. Pourquoi l’humain doit-il rester dans la boucle ?
**Réponse :**  
Pour valider, corriger et contextualiser les décisions, en particulier dans les domaines sensibles (santé, finance, justice).

---

# 38. Questions transversales très probables à l’examen

## Q82. Pourquoi dit-on que le deep learning est data-hungry ?
**Réponse :**  
Le grand nombre de paramètres nécessite de vastes volumes de données pour éviter le sur-apprentissage.

---

## Q83. Pourquoi le deep learning a-t-il explosé récemment ?
**Réponse :**
- puissance GPU,
- grandes bases de données,
- avancées algorithmiques (ReLU, ResNet, Transformers).

---

## Q84. Quelle est la place du deep learning en entreprise ?
**Réponse :**  
Il est utilisé lorsque :
- les données sont complexes,
- la performance prime,
- le coût est justifié par la valeur métier.

---

# 39. Questions de synthèse (type 10 points)

## Q85. Expliquez comment choisir une architecture de deep learning en fonction :
- du type de données,
- de la tâche,
- des contraintes industrielles.

**Réponse attendue :**  
Analyse du type de données → choix de l’architecture → évaluation des coûts → validation → déploiement.

---

## Q86. Discutez les limites actuelles du deep learning.
**Réponse attendue :**
- manque d’explicabilité,
- dépendance aux données,
- coûts énergétiques,
- fragilité face aux biais.

---

# 40. Capacité de généralisation et biais–variance

## Q87. Expliquez le compromis biais–variance en deep learning.
**Réponse :**  
Le compromis biais–variance décrit la tension entre :
- **biais élevé** : modèle trop simple, erreurs systématiques (underfitting),
- **variance élevée** : modèle trop complexe, sensible aux données d’entraînement (overfitting).

Le deep learning tend à réduire le biais mais peut augmenter la variance, d’où la nécessité de régularisation.

---

## Q88. Pourquoi un modèle très performant en entraînement peut-il échouer en test ?
**Réponse :**  
Parce qu’il a mémorisé des régularités spécifiques au dataset d’entraînement qui ne se généralisent pas aux nouvelles données.

---

# 41. Taille du modèle et nombre de paramètres

## Q89. Pourquoi le nombre de paramètres est-il critique ?
**Réponse :**  
Un nombre élevé de paramètres :
- augmente la capacité expressive,
- accroît le risque d’overfitting,
- augmente le coût mémoire et calculatoire.

Le choix de la taille du modèle doit être cohérent avec la taille du dataset.

---

## Q90. Pourquoi les CNN ont-ils moins de paramètres que les MLP sur images ?
**Réponse :**  
Grâce au **partage des poids** et à la **localité spatiale**, un même filtre est appliqué sur toute l’image, réduisant drastiquement le nombre de paramètres.

---

# 42. Robustesse et bruit

## Q91. Comment le bruit affecte-t-il l’apprentissage ?
**Réponse :**  
Le bruit peut :
- ralentir la convergence,
- induire des frontières de décision erronées,
- favoriser l’overfitting si le modèle tente de l’expliquer.

---

## Q92. Comment rendre un modèle plus robuste au bruit ?
**Réponse :**
- data augmentation,
- régularisation,
- dropout,
- early stopping,
- augmentation du dataset.

---

# 43. Early stopping

## Q93. Qu’est-ce que l’early stopping ?
**Réponse :**  
L’early stopping consiste à arrêter l’entraînement lorsque la performance sur l’ensemble de validation cesse de s’améliorer, afin d’éviter le sur-apprentissage.

---

## Q94. Pourquoi l’early stopping est-il efficace ?
**Réponse :**  
Il agit comme une régularisation implicite en limitant la complexité effective du modèle.

---

# 44. Validation croisée et deep learning

## Q95. Pourquoi la validation croisée est-elle rarement utilisée en deep learning ?
**Réponse :**  
Elle est très coûteuse en calcul, car chaque pli nécessite un entraînement complet du réseau.

---

## Q96. Comment compenser l’absence de validation croisée ?
**Réponse :**
- séparation train/validation/test rigoureuse,
- monitoring précis,
- répétition d’expériences avec différentes initialisations.

---

# 45. Apprentissage supervisé, non supervisé et auto-supervisé

## Q97. Différence entre apprentissage supervisé et non supervisé.
**Réponse :**
- Supervisé : données annotées.
- Non supervisé : aucune étiquette, découverte de structure.

---

## Q98. Qu’est-ce que l’apprentissage auto-supervisé ?
**Réponse :**  
Il consiste à générer des pseudo-labels à partir des données elles-mêmes (ex. prédiction de parties masquées), permettant d’exploiter de grands volumes de données non annotées.

---

# 46. Modèles de fondation

## Q99. Qu’est-ce qu’un modèle de fondation ?
**Réponse :**  
Un modèle de fondation est un modèle pré-entraîné à très grande échelle sur des données générales, puis adaptable à de nombreuses tâches par fine-tuning.

---

## Q100. Pourquoi les modèles de fondation sont-ils centraux aujourd’hui ?
**Réponse :**
- mutualisation des coûts d’entraînement,
- performances élevées,
- transfert efficace vers des tâches spécifiques.

---

# 47. Transfer learning – approfondissement

## Q101. Pourquoi le transfer learning fonctionne-t-il même avec peu de données ?
**Réponse :**  
Parce que les couches profondes capturent des patterns génériques réutilisables (bords, formes, structures temporelles).

---

## Q102. Quelle est la différence entre feature extraction et fine-tuning ?
**Réponse :**
- Feature extraction : gel complet du réseau pré-entraîné.
- Fine-tuning : dégel partiel ou total avec faible learning rate.

---

# 48. Problèmes numériques et stabilité

## Q103. Qu’est-ce que l’explosion du gradient ?
**Réponse :**  
Les gradients deviennent très grands, entraînant des mises à jour instables et une divergence de l’apprentissage.

---

## Q104. Qu’est-ce que la disparition du gradient ?
**Réponse :**  
Les gradients deviennent quasi nuls dans les couches profondes, empêchant l’apprentissage.

---

## Q105. Comment atténuer ces problèmes ?
**Réponse :**
- ReLU et variantes,
- normalisation (batch normalization),
- LSTM/GRU,
- skip connections.

---

# 49. Batch normalization

## Q106. Quel est le rôle de la batch normalization ?
**Réponse :**  
Elle normalise les activations intermédiaires afin de stabiliser et accélérer l’apprentissage.

---

## Q107. Pourquoi la batch normalization agit-elle comme une régularisation ?
**Réponse :**  
Elle introduit un bruit dépendant du batch, réduisant l’overfitting.

---

# 50. Comparaison CNN / Transformers (vision)

## Q108. Pourquoi les Transformers sont-ils utilisés en vision ?
**Réponse :**  
Ils permettent de modéliser des dépendances globales dans l’image, là où les CNN sont plus locaux.

---

## Q109. CNN ou Transformer : lequel choisir ?
**Réponse :**  
- CNN : datasets moyens, efficacité, inductive bias fort.
- Transformers : grands datasets, flexibilité, puissance globale.

---

# 51. Évaluation qualitative vs quantitative

## Q110. Pourquoi l’évaluation qualitative est-elle indispensable en génération ?
**Réponse :**  
Les métriques numériques capturent mal la perception humaine (réalisme, diversité, cohérence).

---

## Q111. Pourquoi faut-il combiner plusieurs métriques ?
**Réponse :**  
Aucune métrique unique ne capture toutes les dimensions de la performance.

---

# 52. Explicabilité – limites fondamentales

## Q112. Pourquoi Grad-CAM n’explique-t-il pas réellement la décision ?
**Réponse :**  
Il indique quelles régions ont contribué à la décision, mais pas la chaîne causale exacte.

---

## Q113. Pourquoi l’interprétation humaine peut être biaisée ?
**Réponse :**  
L’humain projette du sens (objets, concepts) là où le réseau manipule uniquement des corrélations statistiques.

---

# 53. Deep learning et causalité

## Q114. Pourquoi le deep learning n’est-il pas causal ?
**Réponse :**  
Il apprend des corrélations statistiques sans modéliser explicitement les relations de cause à effet.

---

## Q115. Pourquoi est-ce problématique ?
**Réponse :**  
Un changement de contexte peut invalider les prédictions, même si les corrélations observées disparaissent.

---

# 54. Contraintes industrielles

## Q116. Pourquoi un modèle très performant peut être inutilisable en production ?
**Réponse :**
- latence excessive,
- coût matériel,
- difficulté de maintenance,
- contraintes réglementaires.

---

## Q117. Quels compromis sont faits en entreprise ?
**Réponse :**
- modèles plus simples,
- compression (quantization, pruning),
- précision légèrement sacrifiée au profit de la robustesse.

---

# 55. Questions de synthèse “excellente copie”

## Q118. Discutez le rôle du deep learning dans la chaîne de valeur d’un projet data.
**Réponse attendue :**  
Positionnement du modèle comme un composant parmi les données, l’évaluation, le déploiement et la décision métier.

---

## Q119. Expliquez pourquoi le deep learning ne remplace pas l’expertise humaine.
**Réponse attendue :**
- absence de compréhension sémantique,
- dépendance aux données,
- nécessité de validation et de contextualisation.

---

## Q120. Donnez une vision critique de l’avenir du deep learning.
**Réponse attendue :**
- modèles plus efficaces,
- meilleure explicabilité,
- hybridation avec raisonnement symbolique,
- contraintes énergétiques croissantes.

---

# 56. Hypothèses inductives et inductive bias

## Q121. Qu’est-ce qu’un inductive bias en deep learning ?
**Réponse :**  
Un inductive bias est un ensemble d’hypothèses intégrées implicitement dans un modèle qui guide l’apprentissage.  
Exemples :
- CNN : hypothèse de localité spatiale et invariance par translation,
- RNN/LSTM : dépendance temporelle,
- Transformers : dépendances globales via attention.

Ces biais facilitent l’apprentissage en réduisant l’espace des solutions possibles.

---

## Q122. Pourquoi les inductive bias sont-ils nécessaires ?
**Réponse :**  
Sans inductive bias, l’apprentissage nécessiterait des volumes de données irréalistes.  
Ils permettent au modèle de généraliser à partir d’un nombre fini d’exemples.

---

# 57. Théorie de l’approximation universelle

## Q123. Que dit le théorème de l’approximation universelle ?
**Réponse :**  
Un réseau de neurones avec une seule couche cachée et suffisamment de neurones peut approximer toute fonction continue sur un compact, sous certaines conditions.

---

## Q124. Pourquoi ce théorème ne suffit-il pas en pratique ?
**Réponse :**  
Parce qu’il ne dit rien :
- sur la taille nécessaire du réseau,
- sur la faisabilité de l’apprentissage,
- sur la généralisation.

La profondeur permet une représentation plus efficace.

---

# 58. Apprentissage profond vs apprentissage classique

## Q125. Pourquoi les méthodes classiques restent-elles pertinentes ?
**Réponse :**  
Elles sont :
- plus interprétables,
- moins coûteuses,
- efficaces sur petits datasets,
- plus simples à maintenir.

---

## Q126. Quand le deep learning devient-il indispensable ?
**Réponse :**  
Lorsque les données sont :
- non structurées (images, texte, audio),
- très volumineuses,
- de forte complexité.

---

# 59. Données manquantes et bruitées

## Q127. Comment gérer les données manquantes en deep learning ?
**Réponse :**
- imputation simple (moyenne, médiane),
- modèles dédiés,
- suppression si marginale.

Le deep learning ne traite pas nativement les valeurs manquantes.

---

## Q128. Pourquoi le bruit d’annotation est-il problématique ?
**Réponse :**  
Le modèle apprend des erreurs comme si elles étaient vraies, ce qui dégrade la généralisation.

---

# 60. Apprentissage multi-tâches

## Q129. Qu’est-ce que l’apprentissage multi-tâches ?
**Réponse :**  
Il consiste à entraîner un même modèle sur plusieurs tâches simultanément, partageant une partie des représentations.

---

## Q130. Quels sont ses avantages ?
**Réponse :**
- meilleure généralisation,
- régularisation implicite,
- exploitation de tâches corrélées.

---

# 61. Modèles séquentiels avancés

## Q131. Pourquoi les Transformers n’utilisent-ils pas de récurrence ?
**Réponse :**  
La récurrence empêche la parallélisation.  
L’attention permet de traiter la séquence en une seule passe.

---

## Q132. Comment les Transformers gèrent-ils l’ordre des séquences ?
**Réponse :**  
Via des **encodages positionnels** ajoutés aux embeddings.

---

# 62. Complexité algorithmique

## Q133. Quelle est la complexité d’un mécanisme d’attention standard ?
**Réponse :**  
Elle est quadratique en fonction de la longueur de la séquence, ce qui limite les contextes très longs.

---

## Q134. Pourquoi cherche-t-on des variantes d’attention ?
**Réponse :**  
Pour réduire la complexité mémoire et permettre le traitement de longues séquences.

---

# 63. Robustesse et attaques adversariales

## Q135. Qu’est-ce qu’une attaque adversariale ?
**Réponse :**  
Une modification imperceptible des données d’entrée entraînant une prédiction erronée du modèle.

---

## Q136. Pourquoi les réseaux sont-ils vulnérables ?
**Réponse :**  
Ils apprennent des frontières de décision complexes et non causales, sensibles à de petites perturbations.

---

# 64. Compression de modèles

## Q137. Pourquoi compresser un modèle deep learning ?
**Réponse :**
- déploiement sur systèmes embarqués,
- réduction de la latence,
- diminution de la consommation énergétique.

---

## Q138. Quelles techniques de compression existent ?
**Réponse :**
- pruning,
- quantization,
- distillation.

---

# 65. Distillation de connaissances

## Q139. Qu’est-ce que la distillation ?
**Réponse :**  
Un modèle léger (student) apprend à imiter un modèle complexe (teacher), en reproduisant ses sorties.

---

## Q140. Pourquoi la distillation fonctionne-t-elle ?
**Réponse :**  
Les sorties du teacher contiennent une information plus riche que les labels bruts (soft targets).

---

# 66. Apprentissage en ligne et incrémental

## Q141. Qu’est-ce que l’apprentissage en ligne ?
**Réponse :**  
Le modèle est mis à jour progressivement à mesure que de nouvelles données arrivent.

---

## Q142. Quels sont les risques de l’apprentissage en ligne ?
**Réponse :**
- oubli catastrophique,
- dérive conceptuelle,
- instabilité.

---

# 67. Continual learning

## Q143. Qu’est-ce que l’oubli catastrophique ?
**Réponse :**  
Le modèle oublie les anciennes tâches lorsqu’il est entraîné sur de nouvelles données.

---

## Q144. Comment limiter l’oubli catastrophique ?
**Réponse :**
- régularisation,
- replay de données,
- architectures modulaires.

---

# 68. Évaluation expérimentale

## Q145. Pourquoi répéter les expériences en deep learning ?
**Réponse :**  
À cause de la variabilité liée à :
- l’initialisation,
- l’ordre des données,
- le stochasticité de l’optimisation.

---

## Q146. Pourquoi fixer une seed n’est-il pas toujours suffisant ?
**Réponse :**  
Certaines opérations GPU sont non déterministes, ce qui introduit une variabilité résiduelle.

---

# 69. Lecture critique des résultats

## Q147. Pourquoi une courbe de loss décroissante ne garantit-elle pas un bon modèle ?
**Réponse :**  
Elle peut masquer :
- un sur-apprentissage,
- une mauvaise généralisation,
- un biais dans les données.

---

## Q148. Pourquoi analyser les erreurs est-il essentiel ?
**Réponse :**  
L’analyse des erreurs permet :
- d’identifier les biais,
- d’améliorer les données,
- de guider les choix architecturaux.

---

# 70. Questions “pièges” fréquentes

## Q149. Un réseau plus profond est-il toujours meilleur ?
**Réponse :**  
Non. Sans données suffisantes ou architecture adaptée, il peut être moins performant.

---

## Q150. Peut-on expliquer exactement une décision d’un réseau profond ?
**Réponse :**  
Non. On peut fournir des indices (saliency, Grad-CAM), mais pas une explication causale exacte.

---

# 71. Questions de conclusion (niveau distinction)

## Q151. Pourquoi le deep learning est-il à la fois puissant et dangereux ?
**Réponse attendue :**
- puissance prédictive élevée,
- opacité,
- dépendance aux données,
- risques sociétaux.

---

## Q152. Quel est le rôle du data scientist face au deep learning ?
**Réponse attendue :**
- concepteur,
- évaluateur critique,
- garant de l’éthique et de la validité.

---

## Q153. Résumez en une page la philosophie du deep learning.
**Réponse attendue :**
- apprentissage de représentations,
- données avant modèles,
- compromis performance / interprétabilité,
- nécessité de contrôle humain.

---

# FIN – BANQUE ABSOLUMENT COMPLÈTE DE QUESTIONS DE COURS
