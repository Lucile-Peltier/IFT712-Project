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
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# Lire les données test et train
test_data = pd.read_csv(os.getcwd() + '/info/test.csv')
train_data = pd.read_csv(os.getcwd() + '/info/train.csv')

algorithme = 'SVM'
ch_hyp = True

#import SVM as alg

if algorithme == 'SVM':
    import SVM
    classif = SVM.SupportVectorMachine()


def main():

    g_donnees = gd.GestionDonnees(train_data, test_data)
    [types, x_tr, t_tr, x_ts] = g_donnees.lecture_donnees(train_data,test_data)
    #x_entr, t_entr, x_val, t_val = gd.split_donnees(x_tr,t_tr,0)
        
    # Entraînement
    #classif = alg.SupportVectorMachine()
    classif.entrainement(x_tr, t_tr, ch_hyp)
    
    # Prédictions pour les ensembles d'entraînement et de test
    predict_tr = classif.prediction(x_tr)
    sc_tr = classif.precision(x_tr, t_tr)
    
    predict_ts = classif.prediction(x_ts)
    sc_ts = classif.precision(x_ts, predict_ts)
    
    return sc_tr, sc_ts

if __name__ == "__main__":
    main()
#sc_train, sc_test = main()

#print('Le score d\'entraînement est :', sc_train)
#print('Le score de validation est :', sc_test)

    