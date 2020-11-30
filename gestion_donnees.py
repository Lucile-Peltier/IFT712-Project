# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:26:37 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np   # Algébra linéal
import pandas as pd  # Analyse de données
from sklearn.preprocessing import LabelEncoder # Gérer les noms des cibles
from sklearn.model_selection import train_test_split
# rom sklearn.model_selection import KFold



class GestionDonnees :
    def __init__(self, bd) :
        #self.donnees_base = donnees_base
        self.bd = bd
        
    def lecture_donnees(self, base_d) :
        """
        Paramètres
        ----------
        base_d :  Numpy array
                  Matrice  lu du fichier base avec l'ensemble de données d'entraînement.

        Returns
        -------
        f_types :   list
                    List avec les noms de types de feuilles.
        x_base :   Numpy array
                    Contient uniquement les données d'entraînement.
        t_base :   Numpy array
                    Contient uniquement les cibles (chiffres) pour entraînement.

        """
        # Lire les types des feuilles du fichier train
        encoder = LabelEncoder().fit(base_d.species)
        f_types = list(encoder.classes_)
        t_base = encoder.transform(base_d.species)
        
        # Séparer id de l'ensemble de données
        x_base_df = base_d.drop(['id', 'species'], axis= 1 )
        x_base = x_base_df.to_numpy()
        
        return f_types, x_base, t_base
    
    def sep_donnees(self, x_data,t_data) :
        
        # Separer les données pour entraînement et test
        x_tr, x_te, t_tr, t_te = train_test_split(x_data, t_data, test_size = 0.2)
                
        return x_tr, x_te, t_tr, t_te
            



