# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:30:09 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV, KFold
# import matplotlib.pyplot as plt


class ArbreDecision:
    def __init__(self):
        """
        Algorithme d'arbre de décision
        
        """
        self.prof_max = 40 # Profondeur maximale du l'arbre
        self.msf = 3  # Nombre minimal de samples dans une feuille
        self.mfn = 100 # Nombre maximal de nodes de feuilles
    
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour l'Arbre de décisions
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement

        Méthode de Grid Search: 
            prof_max: Profondeur maximale entre 5 et 50
            msf: Nombre minimal de samples dans une feuille entre 3 et 10
            Mesure de la qualité de la séparation: giny et entropy
        
        Retourne une dictionaire avec les meilleurs hyperparamètres
        """
        valeurs_prof = np.linspace(5,50,5)
        valeurs_msf = np.linspace(2,10,5, dtype='int')
        p_grid = [{'criterion': ['gini'], 'max_depth': valeurs_prof, \
                   'min_samples_leaf': valeurs_msf, 'max_leaf_nodes': [self.mfn]},
                  {'criterion': ['entropy'], 'max_depth': valeurs_prof, \
                   'min_samples_leaf': valeurs_msf, 'max_leaf_nodes': [self.mfn]}]
        
        cross_v = KFold(10, True) # Cross-Validation
            
        # Recherche d'hyperparamètres
        self.classif = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=p_grid, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec SVM
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleur type de noyau et ses hyperparamètres
        
        Retourne objet avec le modèle entraîné
        """

        if cherche_hyp == True:
            print('Debut de l\'entrainement avec recherche d\'hyperparamètres')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement sans recherche d\'hyperparamètres')
            parametres = {'criterion': 'gini', 'max_depth': self.prof_max, \
                   'min_samples_leaf': self.msf, 'max_leaf_nodes': self.mfn}
            
        self.classif = tree.DecisionTreeClassifier(**parametres)
        print('Fin de l\'entrainement')
        
        #arbre_fin = self.classif.fit(x_train, t_train)
        #tree.plot_tree(arbre_fin)
        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec SVM
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p
    
    def precision(self, x, t):
        """
        Précision ou score du modèle SVM
        
        x = Numpy array avec données de test
        t = Numpy array avec les cibles de class
        
        Retourne le score
        """
        return self.classif.score(x, t)
    
    