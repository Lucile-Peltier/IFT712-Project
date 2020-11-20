# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:27:11 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np
import pandas as pd
import os
import gestion_donnees as gd
from warnings import simplefilter

import SVM
import arbre_decision

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Lire les données test et train
test_data = pd.read_csv(os.getcwd() + '/info/test.csv')
train_data = pd.read_csv(os.getcwd() + '/info/train.csv')

algorithme = 'Arbre_decision'
ch_hyp = True


if algorithme == 'Perceptron':
    print('Ce méthode n\'est pas prêt encore')
    
elif algorithme == 'SVM':
    classif = SVM.SupportVectorMachine()
    
elif algorithme == 'Proches_voisins': 
    print('Ce méthode n\'est pas prêt encore')
    
elif algorithme == 'Naive_Bayesienne': 
    print('Ce méthode n\'est pas prêt encore')

elif algorithme == 'Arbre_decision': 
    classif = arbre_decision.ArbreDecision()

elif algorithme == 'Reseau_neurones': 
    print('Ce méthode n\'est pas prêt encore')

def main():

    g_donnees = gd.GestionDonnees(train_data, test_data)
    [types, x_tr, t_tr, x_ts] = g_donnees.lecture_donnees(train_data,test_data)
        
    # Entraînement
    classif.entrainement(x_tr, t_tr, ch_hyp)
    
    # Prédictions pour les ensembles d'entraînement et de test
    predict_tr = classif.prediction(x_tr)
    sc_tr = classif.precision(x_tr, t_tr)
    print('Le score d\'entraînement est : ', sc_tr)
    
    predict_ts = classif.prediction(x_ts)
    sc_ts = classif.precision(x_ts, predict_ts)
    
    return sc_tr, sc_ts

if __name__ == "__main__":
    main()

    