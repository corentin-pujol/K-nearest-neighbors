# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 18:47:59 2021

@author: coco8
"""

import random
import math as m
import csv
import matplotlib.pyplot as plt

#Classe Individu afin de récupérer et stocker les différents points du dataset concernant les fleurs
class Individu:
    def __init__(self, a, b, c, d, e, f, label):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.label = label
    def __str__(self):
        return f"({self.a}, {self.b}, {self.c}, {self.d}, {self.e}, {self.f}, {self.label})"
    def __eq__(self, autreindividu): #permet le testd'égalité de deux sepales
        result = False
        if((self.a==autreindividu.a) and (self.b==autreindividu.b) and (self.c==autreindividu.c) and (self.d==autreindividu.d) and (self.e==autreindividu.e) and (self.f==autreindividu.f) and (self.label==autreindividu.label)):
            result = True
        return result
    def __contains__(self,x):
        return True
    def __hash__(self):
        return hash((self.a, self.b, self.c,self.d,self.e,self.f,self.label))
    
def ListeIndividus():
    cr = csv.reader(open("preTest.csv","r")) #Lecture du fichier csv
    liste_indiv = []
    for row in cr: #boucle sur les lignes du fichier
        if(len(row)==7):
            indiv = Individu(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), row[6])
            liste_indiv.append(indiv)
    return liste_indiv

population = ListeIndividus()
#print(len(population)) #Il y a 803 individus dans le fichier csv 

def regLin(x, y): #https://gsalvatovallverdu.gitlab.io/python/moindres-carres/ Lien du site qui m'a permis de trouver les coefficients de la droite de régression
    """
    Ajuste une droite d'équation a*x + b sur les points (x, y) par la méthode
    des moindres carrés.

    Args :
        * x (list): valeurs de x
        * y (list): valeurs de y

    Return:
        * a (float): pente de la droite
        * b (float): ordonnée à l'origine
    """
    # initialisation des sommes
    x_sum = 0.
    x2_sum = 0.
    y_sum = 0.
    xy_sum = 0.
    # calcul des sommes
    for xi, yi in zip(x, y):
        x_sum += xi
        x2_sum += xi**2
        y_sum += yi
        xy_sum += xi * yi
    # nombre de points
    npoints = len(x)
    # calcul des paramétras
    a = (npoints * xy_sum - x_sum * y_sum) / (npoints * x2_sum - x_sum**2)
    b = (x2_sum * y_sum - x_sum * xy_sum) / (npoints * x2_sum - x_sum**2)
    # renvoie des parametres
    return a, b

from scipy import stats

def AnalyseGraphiques(population): #https://mrmint.fr/regression-lineaire-python-pratique lien du site qui m'a permis de tracer la droite de régression
      
    Ax=[] #Coordonnées des points de classe A
    Ay=[]
    Bx=[] #Coordonnées des points de classe B
    By=[]
    X=[] #Coordonnées des points des deux classes, afin de tracer la droite de régression linéaire afin de supprimer les points abérrents provoquant des erreurs de prédiction
    Y=[]
    for e in population:
        if(e.label=="classA"):
            Ax.append(e.a)
            X.append(e.a)
            Ay.append(e.f)
            Y.append(e.f)
        else:
            if(e.label=="classB"):
                Bx.append(e.a)
                X.append(e.a)
                By.append(e.f)
                Y.append(e.f)
     
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)    
    fitLine = []
    for a in X:
        fitLine.append(slope*a+intercept)        
    plt.scatter(Ax,Ay,color='b')
    plt.scatter(Bx,By,color='g')
    plt.plot(X, fitLine, c='r')
    plt.xlabel('a')
    plt.ylabel('f')
    plt.show()
    #print(regLin(X, fitLine)) #Coeffcient de la droite de régression

#AnalyseGraphiques(population)

def NettoyagePopulation(population):
    for e in population:
        if(e.label=="classA"):
            if(e.f + 0.02455590941138824*e.a - 0.63379220069064224<0): #On vérifie si le point de classe A est en dessous de la droite de régression
                population.remove(e)
        else:
            if(e.label=="classB"):
                if(e.f + 0.02455590941138824*e.a - 0.63379220069064224>0): #On vérifie si le point de classe B est en dessous de la droite de régression
                    population.remove(e)
    return population

population = NettoyagePopulation(population)
#nalyseGraphiques(population)
#print(len(population))

def ListeTraining(population):
    liste_training = []
    cpt = 0
    while(cpt!=int(80*len(population)/100)): #*On prend 80% du set de données pour la liste d'entrainement
        index_random = random.randint(0,len(population)-1)
        
        while(population[index_random] in liste_training): #Si l'élément est deja dans la liste on change d'index jusqu'à trouver un élément à ajouter
            index_random = random.randint(0,len(population)-1)
            
        liste_training.append(population[index_random])        
        cpt += 1 
        
    return liste_training

listeTraining=ListeTraining(population)

def ListeTest(population,liste_training):
    liste_test = []
    for e in population:
            if(e not in liste_training):
                liste_test.append(e)        
    return liste_test

listeTest = ListeTest(population, listeTraining)

#print(len(listeTraining),len(listeTest))

def ListeIndividus_FinalTest():
    cr = csv.reader(open("finalTest.csv","r")) #Lecture du fichier csv
    liste_indiv = []
    for row in cr: #boucle sur les lignes du fichier
        if(len(row)==6):
            indiv = Individu(float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]), "classInconnue")
            liste_indiv.append(indiv)
    return liste_indiv

finalTest_population = ListeIndividus_FinalTest()


def Distance(individu1, individu2):
    d = m.sqrt((float(individu1.a)-float(individu2.a))**2+(float(individu1.b)-float(individu2.b))**2+(float(individu1.c)-float(individu2.c))**2+(float(individu1.d)-float(individu2.d))**2+(float(individu1.e)-float(individu2.e))**2+(float(individu1.f)-float(individu2.f))**2)
    return d

def Dico_k_erreur_init(population):
    dico = {}
    for i in range(len(population)):
        dico[i] = 0
    return dico
    
def Prediction_label(liste_voisins):
    nb_A=0
    nb_B=0
    nb_C=0
    nb_D=0
    nb_E=0
    for i in range(len(liste_voisins)):
        #print(liste_voisins[i].label)
        if(liste_voisins[i].label=="classA"):
            nb_A += 1
        elif(liste_voisins[i].label=="classB"):
            nb_B += 1
        elif(liste_voisins[i].label=="classC"):
            nb_C += 1
        elif(liste_voisins[i].label=="classD"):
            nb_D += 1
        else:
            nb_E += 1
            
    dico={"classA":nb_A, "classB":nb_B, "classC":nb_C, "classD":nb_D, "classE":nb_E};
    #print(dico)
    liste_triee = sorted(dico.items(), key=lambda x: x[1])
    #print(liste_triee)
    #print(liste_triee[len(liste_triee)-1][0])
    return liste_triee[len(liste_triee)-1][0]
    
            
            
def K_nn(liste_training,liste_test,k):
    dico_prediction_test={}
    for e in liste_test:
        dico_voisins_distance={}
        for a in liste_training:
            distance = Distance(e,a)
            dico_voisins_distance[a] = distance
        cpt=1
        k_voisins=[]
        for key, value in sorted(dico_voisins_distance.items(), key=lambda x: x[1]):
            if(cpt<=k):
                k_voisins.append(key)
            cpt += 1            
        label=Prediction_label(k_voisins)
        dico_prediction_test[e] = label
        #print(e,dico_prediction_test[e])
    return dico_prediction_test    


def TauxErreur(dico_prediction):
    nb_erreur = 0
    for k, v in dico_prediction.items():
        if(k.label!=v):
            nb_erreur += 1
    return nb_erreur     

def RechercheBestK(liste_training,liste_test):
    dico_k_erreur = {}
    for k in range(2,25):
        dico_pred = K_nn(liste_training,liste_test,k)
        taux_erreur = TauxErreur(dico_pred)*100/len(listeTest)
        dico_k_erreur[k] = taux_erreur
    print(sorted(dico_k_erreur.items(), key=lambda x: x[1]))
    for k, v in sorted(dico_k_erreur.items(), key=lambda x: x[1]):
        #print(k,v)
        return k

"""
meilleur_k = RechercheBestK(listeTraining,listeTest)
dico_pred = K_nn(listeTraining,listeTest,meilleur_k)
#for k, v in dico_pred.items():
    #print(k,v)
print(meilleur_k,str(TauxErreur(dico_pred)*100/len(listeTest)) + "% d'erreurs")
"""


"""           
cpt=0
fichier = open("resultats_test_Knn_RégressionAB_preTest.txt","w")
while(cpt!=100):
    listeTraining=ListeTraining(population)
    listeTest = ListeTest(population, listeTraining)
    k = 0
    k = RechercheBestK(listeTraining,listeTest)
    dico_pred = {}
    dico_pred = K_nn(listeTraining,listeTest,k)
    print(k,str(TauxErreur(dico_pred)*100/len(listeTest)) + "% d'erreurs")
    fichier.write(str(k) + " => "  + str(TauxErreur(dico_pred)*100/len(listeTest)) + "% d'erreurs\n")
    cpt+=1
fichier.close()


dico_pred = K_nn(population,finalTest_population,3)
fichier = open("PUJOL_Sample_finalTest.txt","w")
for k, v in dico_pred.items():
    k.label = v
for e in finalTest_population:
    for k, v in dico_pred.items():
        if((e.a==k.a) and (e.b==k.b) and (e.c==k.c) and (e.d==k.d) and (e.e==k.e) and (e.f==k.f)):
            fichier.write(str(k.label)+"\n")
fichier.close()
"""