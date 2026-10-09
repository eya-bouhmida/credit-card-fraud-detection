# Détection de fraude bancaire par Machine Learning

**Détecter des transactions frauduleuses très rares (0,52 %) parmi 1,85 million de transactions par carte bancaire**

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-2a78d6)
![SHAP](https://img.shields.io/badge/SHAP-Explicabilit%C3%A9-eb6834)

| Meilleur rappel | Meilleur F1 | Meilleure AUC-PR | Données |
|:---:|:---:|:---:|:---:|
| **97,3 %** des fraudes détectées (XGBoost) | **0,62** (Random Forest) | **0,918** (XGBoost) | 1 852 394 transactions, dont 9 651 fraudes |

---

## Problématique

La fraude par carte bancaire représente une infime partie des transactions, mais chaque fraude non détectée est une perte directe. Le défi est double :

- **Le déséquilibre extrême des classes** : avec 0,52 % de fraudes, un modèle qui prédit toujours « légitime » obtient 99,5 % d'accuracy tout en ne détectant aucune fraude. L'accuracy est donc inutilisable comme métrique.
- **Le compromis métier** : détecter plus de fraudes (rappel) génère plus de fausses alertes (précision), qui ont elles aussi un coût pour la banque et ses clients.

## Dataset

[Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (Kaggle, données simulées).

| | |
|---|---|
| Transactions | 1 852 394 |
| Fraudes | 9 651 (**0,52 %**) |
| Variables d'origine | 23 (montant, catégorie de marchand, date et heure, client, localisation…) |
| Valeurs manquantes / doublons | Aucun |

**Principaux constats de l'exploration :**
- Le **montant médian** d'une fraude est de **390 $**, contre **47 $** pour une transaction légitime.
- Le taux de fraude varie fortement selon la **catégorie de marchand** : 1,6 % pour `shopping_net`, 1,3 % pour `misc_net` et `grocery_pos`, contre moins de 0,2 % pour `home` ou `health_fitness`.
- La **distance** entre le client et le marchand ne distingue pas les fraudes (médiane de 78 km dans les deux cas).

## Méthodologie

| Étape | Contenu |
|---|---|
| **1. Nettoyage** | Suppression des identifiants (numéro de carte, nom, numéro de transaction) pour que le modèle apprenne des comportements et non des identités |
| **2. Feature engineering** | Âge du client, heure, jour de la semaine, week-end, distance client-marchand, `log(montant + 1)` |
| **3. Sélection des variables** | Khi-2 et V de Cramér (variables qualitatives), ANOVA (variables quantitatives), VIF (multicolinéarité). Distance et population de la ville rejetées (p > 0,6). **9 variables retenues** |
| **4. Découpage** | Train / test 80 / 20 **stratifié** (0,5 % de fraudes dans chaque ensemble) |
| **5. Rééquilibrage** | **SMOTE** appliqué **uniquement sur le train** (passage de 0,5 % à 50 % de fraudes) ; le test reste à la proportion réelle |
| **6. Modélisation** | Régression logistique, AFD, arbre de décision, Random Forest, XGBoost (pondération des classes `scale_pos_weight = 190,9`) |
| **7. Optimisation** | `RandomizedSearchCV` sur le Random Forest |
| **8. Explicabilité** | SHAP : importance globale, bee swarm et explication d'une transaction individuelle |

## Résultats

Évaluation sur le jeu de test (370 479 transactions, dont 1 930 fraudes), à la proportion réelle de fraudes.

| Modèle | Rappel | Précision | F1 | AUC-ROC | AUC-PR |
|---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** | **0,973** | 0,307 | 0,466 | **0,999** | **0,918** |
| **Random Forest** | 0,930 | **0,467** | **0,622** | 0,997 | 0,882 |
| Random Forest optimisé | 0,942 | 0,377 | 0,539 | 0,997 | 0,881 |
| Arbre de décision | 0,959 | 0,171 | 0,290 | 0,993 | 0,678 |
| Régression logistique | 0,780 | 0,022 | 0,042 | 0,842 | 0,160 |
| AFD | 0,780 | 0,021 | 0,041 | 0,842 | 0,158 |

**Lecture :**
- **L'AUC-PR est la métrique la plus fiable ici.** L'AUC-ROC dépasse 0,99 pour tous les modèles à base d'arbres, car elle est gonflée par l'énorme majorité de transactions légitimes. L'AUC-PR, elle, sépare nettement les modèles (de 0,16 à 0,92), pour une valeur de référence de 0,005.
- **XGBoost** détecte le plus de fraudes (97,3 %), mais environ 7 alertes sur 10 sont de fausses alarmes.
- **Random Forest** offre le meilleur équilibre (F1 = 0,62) : environ une alerte sur deux est une vraie fraude.
- **Les modèles linéaires** (régression logistique, AFD) sont insuffisants : la relation entre les variables et la fraude n'est pas linéaire.
- **L'optimisation des hyperparamètres** a augmenté le rappel (0,930 → 0,942) mais dégradé le F1 (0,622 → 0,539) : la recherche a été guidée par un F1 calculé sur des données rééquilibrées par SMOTE, qui ne reflète pas la proportion réelle de fraudes.

### Explicabilité (SHAP)

Trois variables expliquent l'essentiel des prédictions :
1. **Le montant** (`log_amt`) : de loin le signal le plus fort ; un montant élevé pousse vers la fraude.
2. **L'heure de la transaction**.
3. **La catégorie de marchand** : certaines catégories (achats en ligne notamment) sont nettement plus risquées.

Le sexe, l'État, la profession et le week-end ont une contribution quasi nulle et pourraient être retirés d'une version allégée du modèle.

## Recommandation

Le choix du modèle dépend du coût métier :
- **Priorité à la détection** (bloquer un maximum de fraudes) : **XGBoost**.
- **Priorité à la limitation des fausses alertes** (expérience client) : **Random Forest**.

## Limitations

- **Données simulées** : les comportements frauduleux réels sont plus variés et évoluent dans le temps.
- **Précision faible** : même le meilleur modèle génère beaucoup de fausses alertes. Le **seuil de décision** n'a pas été ajusté (seuil par défaut de 0,5), alors qu'un seuil adapté au coût métier améliorerait nettement le compromis.
- **Optimisation biaisée** : la validation croisée a été faite sur des données rééquilibrées par SMOTE, ce qui surestime le F1 (0,99 en validation contre 0,54 en test).
- **Pas de dimension temporelle** : le découpage est aléatoire ; un découpage chronologique serait plus réaliste pour un système en production.

## Structure

```
credit-card-fraud-detection/
├── Fraud_Detection.ipynb   ← notebook complet (EDA, modélisation, évaluation, SHAP)
└── README.md
```

## Reproduire

1. Télécharger le dataset depuis [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv` et `fraudTest.csv`).
2. Ouvrir le notebook dans Google Colab ou Jupyter.
3. Installer les dépendances : `pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib seaborn statsmodels`.

---

**Auteure :** Eya Bouhmida · [GitHub](https://github.com/eya-bouhmida)
