# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 13:26:37 2020

@author: lucile.peltier
         sergio.redondo
"""

import numpy as np   # Algébra linéal
import pandas as pd  # Analyse de données
import os            # Se communiquer avec le système opérative  
# import matplotlib.pyplot as plt
import cv2           # Gérer les images
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold

# Définir les marqueurs du classe et les dossiers avec les images
f_types = os.listdir(os.getcwd() + '/flowers')
f_directories = [os.getcwd() + '\\flowers\\' + it for it in f_types]

class GestionDonnees :
    def __init__(self, donnees_base, d_train, d_test) :
        self.donnees_base = donnees_base
        self.d_test = d_test
        self.d_train = d_train
        
    def lecture_donnees(types, directories) :
        x_base = []
        t_base = []
        list_typ = list(range(len(f_types)))
        for itr in list_typ :
            f_type = f_types[itr]
            path = f_directories[itr]
            im_list = os.listdir(path)
            for im_id in im_list :
                path_com = path + "\\" + im_id
                img = cv2.imread(path_com)
                img_n = cv2.resize(img, (150, 150))
                x_base.append(np.array(img_n))
                t_base.append(str(f_type))
            pass
        return x_base, t_base
                
[xx, tt] = GestionDonnees.lecture_donnees(f_types, f_directories)

print(xx, '\n')
print(tt)