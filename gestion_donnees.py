# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:26:37 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np   # Algébra linéal
import pandas as pd  # Analyse de données
import os            # Se communiquer avec le système opérative  
from sklearn.preprocessing import LabelEncoder # Gérer les noms des cibles

# import matplotlib.pyplot as plt
# import cv2           # Gérer les images
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Lire les données test et train
test_data = pd.read_csv(os.getcwd() + '/info/test.csv')
train_data = pd.read_csv(os.getcwd() + '/info/train.csv')


class GestionDonnees :
    def __init__(self, donnees_base, d_train, d_test) :
        self.donnees_base = donnees_base
        self.d_test = d_test
        self.d_train = d_train
        
    def lecture_donnees(test, train) :
        """
        Parameters
        ----------
        test :  Numpy array
                Matrice lu du fichier base avec l'ensemble de données de test.
        train : Numpy array
                Matrice  lu du fichier base avec l'ensemble de données d'entraînement.

        Returns
        -------
        f_types :   list
                    List avec les noms de types de feuilles.
        x_train :   DataFrame
                    Contient uniquement les données d'entraînement.
        id_tr :     Series (objet panda)
                    Contient les id de l'ensemble d'entraînement.
        t_train :   DataFrame
                    Contient uniquement les cibles (chiffres) pour entraînement.
        x_test :    DataFrame
                    Contient uniquement les données de test.
        id_te :     Series (objet panda)
                    Contient les id de l'ensemble de test.

        """
        # Lire les types des feuilles du fichier train
        encoder = LabelEncoder().fit(train.species)
        f_types = list(encoder.classes_)
        t_train = encoder.transform(train.species)
        
        # Séparer id du train et du test
        id_tr = train.id
        id_te = test.id
        x_train = train.drop(['id', 'species'], axis= 1 )
        x_test = test.drop(['id'], axis = 1)
        
        return f_types, x_train, id_tr, t_train, x_test, id_te
                        
    def split_donnees(x_data, t_data, methode) :
        # Separer les données pour entraînement et validation
        
        x_array = np.array(x_data)
        t_array  = np.array(t_data)
        x_entr = []
        t_entr = []
        x_valid = []
        t_valid = []
        if methode==0 :
            kf = KFold(10, True)
            for idx_entr, idx_test in kf.split(x_data) :
                xen = x_array[idx_entr]
                ten = t_array[idx_entr]
                xva = x_array[idx_test]
                tva = t_array[idx_test]
                x_entr.append(xen)
                t_entr.append(ten)
                x_valid.append(xva)
                t_valid.append(tva)
      
        else :
            x_entr, x_valid, t_entr, t_valid = \
                train_test_split(x_data, t_data, test_size = 0.25)
                
        return x_entr, t_entr, x_valid, t_valid
            
feuilles, xx, i_x, tt, xtst, i_t = GestionDonnees.lecture_donnees(test_data, \
                                                                  train_data)
xetr, tetr, xva, tva = GestionDonnees.split_donnees(xx,tt,0)

print('Finished')
