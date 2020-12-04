# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:27:11 2020

@author: lucile.peltier
         sergio.redondo
"""

# Importer outils générales
import numpy as np
import pandas as pd
import os
import time
import sklearn.metrics

# Importer codes spécifiques
import gestion_donnees as gd
import SVM
import arbre_decision
import foret_aleatoire
import naive_bayesienne


# Ignorer les warnings
from warnings import simplefilter
simplefilter(action='ignore')

# Lire la base de données
d_base = pd.read_csv(os.getcwd() + '/info/train.csv')

algorithme = 'Foret_Aleatoire'
ch_hyp = False

#Importer l'algorithme correspondant

if algorithme == 'Perceptron':
    print('Ce méthode n\'est pas prêt encore')
    
elif algorithme == 'SVM':
    classif = SVM.SupportVectorMachine()
    
elif algorithme == 'Proches_voisins': 
    print('Ce méthode n\'est pas prêt encore')
    
elif algorithme == 'Naive_Bayesienne': 
    classif = naive_bayesienne.NaiveBayes()
    
elif algorithme == 'Arbre_decisions': 
    classif = arbre_decision.ArbreDecision()
    
elif algorithme == 'Foret_Aleatoire': 
    classif = foret_aleatoire.ForetAleatoire()

def main():
    
    # Séparer les données et leur cibles
    g_donnees = gd.GestionDonnees(d_base)
    [types, X, t] = g_donnees.lecture_donnees(d_base)
    
    # Séparer les données pour test et train
    x_tr, x_ts, t_tr, t_ts = g_donnees.sep_donnees(X, t)
        
    # Entraînement
    debut_e = time.time()

    classif.entrainement(x_tr, t_tr, ch_hyp)

    fin_e = time.time()
    print('Fin de l\'entrainement. Réalisé en %.2f secondes.'% (fin_e - debut_e))
    
    # Prédictions pour les ensembles d'entraînement et de test
    predict_tr = classif.prediction(x_tr)
    predict_ts = classif.prediction(x_ts)
    
    # Métriques pour évaluer l'entraînement et test
    sc_tr = sklearn.metrics.accuracy_score(t_tr, predict_tr)
    sc_ts = sklearn.metrics.accuracy_score(t_ts, predict_ts)
    print(sc_tr, sc_ts)

    #print(sklearn.metrics.classification_report(t_ts, predict_ts))
    
    
    return sc_tr, sc_ts

if __name__ == "__main__":
    main()

    