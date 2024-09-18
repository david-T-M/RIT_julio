import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance,entropy
import mutual_info as mi

def obtener_distancia(texto_v,hipotesis_v,texto_t,texto_h,b_col,b_index):
    lista_l=[]
    for i in range(len(texto_t)):
        lista=[]
        for j in range(len(texto_h)):
            lista.append(np.linalg.norm(texto_v[i] - hipotesis_v[j]))#*wasserstein_distance(texto_2[i],hipotesis_2[j]))
        lista_l.append(lista)
    df_distEuc=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    df_distEuc=df_distEuc.drop(b_col[1:],axis=1)
    df_distEuc=df_distEuc.drop(b_index[1:],axis=0)
    return df_distEuc

def get_Semantic_info(nlp,vector):
    vec_S=np.zeros(300,)
    x_mean=vector.mean()
    dev=vector.std()
    alpha=x_mean + dev
    for k in range(len(vec_S)):
        if abs(vec_S[k]-vector[k])>=alpha:
            vec_S[k]=vec_S[k]+vector[k]
    return vec_S  

# modifique este con los datos de getsemantic_info
def wasserstein_mutual_infF(texto_v,hipotesis_v,texto_t,texto_h):  
    lista_l=[]
    lista_muinfor=[]   
    for i in range(len(texto_t)):
        lista=[]
        lista_mu=[]
        for j in range(len(texto_h)):
            lista.append(wasserstein_distance(get_Semantic_info(texto_v[i]),get_Semantic_info(hipotesis_v[j])))
            lista_mu.append(mi.mutual_information_2d(np.array(get_Semantic_info(texto_v[i])),np.array(get_Semantic_info(hipotesis_v[j]))))
        lista_l.append(lista)
        lista_muinfor.append(lista_mu)
    DFmearth=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    DFmutual_inf=pd.DataFrame(lista_muinfor,index=texto_t,columns=texto_h)
    return DFmearth,DFmutual_inf

def wasserstein_mutual_inf(texto_v,hipotesis_v,texto_t,texto_h):  
    lista_l=[]
    lista_muinfor=[]   
    for i in range(len(texto_t)):
        lista=[]
        lista_mu=[]
        for j in range(len(texto_h)):
            lista.append(wasserstein_distance(texto_v[i],hipotesis_v[j]))
            lista_mu.append(mi.mutual_information_2d(np.array(texto_v[i]),np.array(hipotesis_v[j])))
        lista_l.append(lista)
        lista_muinfor.append(lista_mu)
    DFmearth=pd.DataFrame(lista_l,index=texto_t,columns=texto_h)
    DFmutual_inf=pd.DataFrame(lista_muinfor,index=texto_t,columns=texto_h)
    return DFmearth,DFmutual_inf

def entropia(X):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probs = [np.mean(X == valor) for valor in set(X)]
    print("Valores para entropia",set(X))
    print("Probabilidades",probs)
    #return round(sum(-p * np.log2(p) for p in probs), 3)
    return entropy(probs,base=2)

def kullback_leibler(X,Y):
    """Devuelve el valor de entropia de una muestra de datos""" 
    probsX = [np.mean(X == valor) for valor in set(X)]
    print("Valores para entropia",set(X))
    print("Probabilidades",probsX)
    probsY = [np.mean(Y == valor) for valor in set(X)]
    print("Valores para entropia",set(X))
    print("Probabilidades",probsY)
    #return round(sum(-p * np.log2(p) for p in probs), 3)
    return entropy(probsY,probsX,base=2)