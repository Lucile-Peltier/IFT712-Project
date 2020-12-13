# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 19:44:30 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
import scikitplot as skplt


class KProchesVoisins:
    def __init__(self):
        """
        algorithme K plus proches voisins
        
        """
        self.n_neighbors = 5      # nombre de voisins
        self.weights = 'distance' # la valeur des poids dépend de la distance entre les points
        self.algorithm='auto'     # l'algorithme le plus pertinant va être utlisé dans la fonction fit()
        self.leaf_size=30         # nombre de feuilles pour les algorithmes BallTree et KDTree pouvant être chosit dans fit() 
        self.p=2                  # paramètre de puissance pour la métrique de Minkowski
        self.metric='minkowski'   # métrique de distance utilisée
        self.metric_params=None   # paramères additionnels
        self.n_jobs=None          # travaux parallèles
        
       
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour les K plus proches voisins'

        x_tr: Numpy array avec données d'entraînement
        t_tr: Numpy array avec cibles pour l'entraînement

        Méthode de Grid Search:
            n_neighbors : nombre de voisins entre 3 et 15
            weights : poids 'distance' ou 'uniformes'
            algorithme : automatique, 'ball tree', 'kd tree', force brute
            leaf_size : taille des feuilles de 10 à 50
            p: paramètre de puissance de Minkowski entre 1 et 5
            métriques : euclienne, minkowski ou manhattan
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
            
        """
        valeurs_voisins = np.arange(3, 15, dtype='int')
        valeurs_feuilles = np.arange(10, 50, dtype='int')
        valeurs_p = np.arange(1, 5, dtype='int')
        p_grid = {'n_neighbors': valeurs_voisins, 'weights': ['distance', 'uniform'], \
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size' : valeurs_feuilles,\
                  'p': valeurs_p, 'metric': ['euclidean', 'minkowski', 'manhattan']}
        
        cross_v = KFold(10, True) #validation croisée
        
        ## Recherche hyperparamètres
        self.classif = RandomizedSearchCV(estimator=KNeighborsClassifier(), \
                                          param_distributions=p_grid, n_iter=20, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entrainement avec les K plus proches voisins

        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné

        """
        if cherche_hyp == True:
            print('Debut de l\'entrainement K voisins avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        
        else:
            print('Debut de l\'entrainement K voisins sans recherche d\'hyperparamètres','\n')
            parametres = {'n_neighbors': self.n_neighbors, 'weights': self.weights, \
                          'algorithm': self.algorithm, 'leaf_size' : self.leaf_size,\
                          'p': self.p, 'metric': self.metric, 'metric_params': self.metric_params,\
                          'n_jobs': self.n_jobs}
            
        self.classif =KNeighborsClassifier(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement K voisins :',\
              self.classif.get_params(),'\n')
            
        return self.classif.fit(x_train, t_train)
        
    
    def prediction(self, x_p):
        """
        Prédiction avec K proches voisins
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    