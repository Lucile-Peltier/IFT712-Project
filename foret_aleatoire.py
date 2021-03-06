# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:21 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold



class ForetAleatoire:
    def __init__(self):
        """
        Algorithme de Forêt Aléatoire
        
        """
        self.n_arbres = 100 # Nombre d arbres à utiliser
        self.prof_max = 30 # Profondeur maximale du l'arbre
        self.msf = 3  # Nombre minimal de samples dans une feuille
        self.mfn = 110 # Nombre maximal de nodes de feuilles
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour la Forêt Aléatoire
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Randomized Search: 
            n_arbres: Nombre d'arbres entre 50 et 200
            prof_max: Profondeur maximale entre 10 et 30
            msf: Nombre minimal d'échantillons dans une feuille entre 2 et 10
            Mesure de la qualité de la séparation: giny et entropy
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
        """
        valeurs_narb = np.arange(50,200, dtype=int)
        valeurs_prof = np.arange(10,30, dtype=int)
        valeurs_msf = np.arange(2,4, dtype=int)
        p_grid = {'n_estimators': valeurs_narb, 'criterion': ['gini','entropy'], \
                   'max_depth': valeurs_prof, 'min_samples_leaf': valeurs_msf, \
                   'max_leaf_nodes': [self.mfn]}
        
        cross_v = KFold(10, True) # validation croisée
            
        # Recherche d'hyperparamètres
        self.classif = RandomizedSearchCV(estimator=RandomForestClassifier(), \
                                          param_distributions=p_grid, n_iter=25, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec Forêt Aléatoire
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non les meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """

        if cherche_hyp == True:
            print('Debut de l\'entrainement FA avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement FA sans recherche d\'hyperparamètres','\n')
            parametres = {'criterion': 'entropy', 'max_depth': self.prof_max, \
                   'min_samples_leaf': self.msf, 'max_leaf_nodes': self.mfn}
            
        self.classif = RandomForestClassifier(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement FA :',\
              self.classif.get_params(),'\n')

        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec Forêt Aléatoire
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    