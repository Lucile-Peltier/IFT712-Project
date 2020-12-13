# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:27:28 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold


class Perceptron:
    def __init__(self):
        """
        Algotithme du perceptron multi-classe

        """
        self.hidden_layer_sizes = (100,) #taille et nombre de couches cachées
        self.activation = 'relu'         #fonction d'activation de la couche cachée
        self.solver = 'adam'             #optimization des poids
        self.alpha = 0.0001              #terme de régularisation
        self.batch_size = 'auto'         #taille des lots pour l'optimisation stochastique
        self.learning_rate = 'constant'  #vitesse d'apprentissage
        self.learning_rate_init = 0.001  #vitesse d'apprentissage initiale
        self.power_t = 0.5               #exposant pour 'sgd'
        self.max_iter = 200              #nombre maximum d'itérations
        self.shuffle = True              #mélange des échantillons pour 'sgd'
        self.random_state = None         #génération du nombre de générations de poids et biais
        self.tol = 1e-4                  #tolérance pour l'optimisation
        self.verbose = False             #affichage des messages à stdout
        self.warm_start = False          #True, utilise la solution précédente de fit pour l'initialisation
        self.momentum = 0.9              #mometum de la descente de gradient
        self.nesterovs_momentum = True   #utilisation du mometum de Nesterov
        self.early_stopping = False      #arrêt prématuré si le score de validation ne s'améliore pas
        self.validation_fraction = 0.1   #proportion de données de validation pour l'arrêt prématuré
        self.beta_1 = 0.9                #taux de décroissance exponentiel du vecteur 1 pour 'adam'
        self.beta_2 = 0.999              #taux de décroissance exponentiel du vectuer 2 pour 'adam'
        self.epsilon = 1e-8              #stabilité numérique pour 'adam'
        self.n_iter_no_change = 10       #nombre maximal d'epochs ne respectant pas l'amélioration 'tol'
        self.max_fun = 15000             #nombre maximal d'appel de la fonction perte
        
    def recherche_hyper(self, x_tr, t_tr):
        """
        Recherche d'hyperparamètres pour le perceptron

        x_tr: Numpy array avec données d'entraînement
        t_tr: Numpy array avec cibles pour l'entraînement

        Méthode de RandomizedSearch
        
        Retourne un dictionnaire avec les meilleurs hyperparamètres
            
        """
        valeurs_max_iter = np.arange(20, 200, 20)
        valeurs_tol = np.arange(0, 2e-4, 0.2e-4)
        valeurs_learning_rate = np.arange(0.5e-4, 2e-3, 0.5e-4)
        valeurs_momentum = np.arange(0, 1, 0.2)
        
        p_grid = {'activation': ['logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], \
                  'batch_size': [64], 'learning_rate': ['constant', 'invscaling', 'adaptative'],\
                  'learning_rate_init': valeurs_learning_rate, 'max_iter': valeurs_max_iter, \
                  'tol': valeurs_tol, 'momentum': valeurs_momentum}
        
        cross_v = KFold(10, True) #validation croisée
        
        ## Recherche hyperparamètres
        self.classif = RandomizedSearchCV(estimator=MLPClassifier(), \
                                         param_distributions=p_grid, n_iter=20, cv=cross_v)
        self.classif.fit(x_tr, t_tr)
        mei_param = self.classif.best_params_
        
        return mei_param
    
    def entrainement(self, x_train, t_train, cherche_hyp):
        """
        Entraînement avec perceptron
        
        x_train: Numpy array avec données d'entraînement
        t_train: Numpy array avec cibles pour l'entraînement
        cherche_hyp: Chercher ou non le meilleures hyperparamètres
        
        Retourne un objet avec le modèle entraîné
        """
        
        
        if cherche_hyp == True:
            print('Debut de l\'entrainement perceptron avec recherche d\'hyperparamètres','\n')
            parametres = self.recherche_hyper(x_train, t_train)
        else:
            print('Debut de l\'entrainement perceptron sans recherche d\'hyperparamètres','\n')
            parametres = {'hidden_layer_sizes': self.hidden_layer_sizes, 'activation': self.activation, \
                          'solver': self.solver, 'alpha': self.alpha, 'batch_size': self.batch_size, \
                          'learning_rate': self.learning_rate, 'learning_rate_init': self.learning_rate_init, \
                          'power_t': self.power_t, 'max_iter': self.max_iter, 'shuffle': self.shuffle, \
                          'random_state': self.random_state, 'tol': self.tol, 'verbose': self.verbose, \
                          'warm_start': self.warm_start, 'momentum': self.momentum, 'nesterovs_momentum': self.nesterovs_momentum, \
                          'early_stopping': self.early_stopping, 'validation_fraction': self.validation_fraction, \
                          'beta_1': self.beta_1, 'beta_2': self.beta_2, 'epsilon': self.epsilon, \
                          'n_iter_no_change': self.n_iter_no_change, 'max_fun': self.max_fun}
            
        self.classif = MLPClassifier(**parametres)
        
        print('Paramètres utilisés pour l\'entraînement perceptron :',\
              self.classif.get_params(),'\n')
            
        return self.classif.fit(x_train, t_train)
    
    def prediction(self, x_p):
        """
        Prédiction avec perceptron
        
        x_p = Numpy array avec données pour trouver la prédiction
        
        Retourne les cibles t_p pour x_p et leur score
        """
        self.t_p = self.classif.predict(x_p)
        return self.t_p