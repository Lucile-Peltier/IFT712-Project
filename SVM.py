# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:29:27 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
# import matplotlib.pyplot as plt


class SupportVectorMachine:
    def __init__(self):
        """
        Algorithme de machines à vecteurs de support
        
        """
        self.lamb = 0.2
        self.noyau = 'rbf'
    
    def recherche_hyper(self, x_tr, t_tr):
        valeurs_lamb = np.linspace(0.000000001,2,5)
        p_grid = [{'kernel': ['rbf'], 'C': valeurs_lamb, 
                       'gamma': ['scale']},
                      {'kernel': ['linear'], 'C': valeurs_lamb}, \
                      {'kernel': ['poly'], 'C': valeurs_lamb, 
                       'degree': np.arange(2,7), 'coef0': np.arange(0,6)},
                       {'kernel': ['sigmoid'], 'C': valeurs_lamb, 
                       'gamma': ['scale']}]
        
        cross_v = KFold(10, True) # Cross-Validation
            
        # Recherche d'hyperparamètres
        self.classif = GridSearchCV(estimator=svm.SVC(), param_grid=p_grid, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec SVM
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Pour chercher ou non le meilleur noyau et ses hyperparamètres
        
        Retourne objet avec le modèle entraîné
        """
        print('Started training')

        if cherche_hyp == True:
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            parametres = {'kernel': self.noyau, 'C': self.lamb, 'gamma': 'auto'}
            
        #self.classif = svm.SVC(kernel='rbf', C= 0.2, gamma= 'auto') # Classificateur
        self.classif = svm.SVC(**parametres)
        print('Finished training')
        
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
    
    def hyperparametres(self, x_en, t_en):
        """
        Recherche d'hyperparamètre lambda du modèle SVM
        
        x = Numpy array avec données de test
        t = Numpy array avec les cibles de class
        
        Retourne le score
        """
    
        