import pandas as pd
import numpy as np
import utils as ut # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import spacy
import mutual_info as mi
import time
from scipy.stats import wasserstein_distance,entropy
import sys
from math import floor
from scipy import spatial

import conceptnet_lite
conceptnet_lite.connect("../OPENAI/data/conceptnet.db")
from conceptnet_lite import Label, edges_for, edges_between

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

relaciones_generales1=["is_a","part_of","used_for", "capable_of", "at_location","etymologically_related_to","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]
relaciones_contextuales1=["is_a","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent","related_to","similar_to"]

relaciones_generales=["is_a","part_of","used_for", "capable_of", "at_location","etymologically_related_to","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]
relaciones_especificas=["is_a","manner_of","has_a","derived_from","has_property","form_of","causes","has_prerequisite","has_subevent","has_first_subevent"]

def bag_of_synonyms(word):
    sinonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name == "synonym":
                if word== e.start.text:
                    sinonimos.add(e.end.text)
                elif word== e.end.text:
                    sinonimos.add(e.start.text)
    except:
        pass
    sinonimos.add(word)
    return sinonimos

def bag_of_antonyms(word):
    antonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in ["antonym","distinc_from"]:
                if word== e.start.text:
                    antonimos.add(e.end.text)
                elif word== e.end.text:
                    antonimos.add(e.start.text)
    except:
        pass
    return antonimos

def bag_of_hyperonyms(word):
    hiperonimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones_generales:
                if word== e.start.text:
                    hiperonimos.add(e.end.text)
    except:
        pass
    return hiperonimos

def bag_of_hyponyms(word):
    hiponimos=set()
    try:
        for e in edges_for(Label.get(text=word, language='en').concepts, same_language=True):
            if e.relation.name in relaciones_especificas:
                if word== e.end.text:
                    hiponimos.add(e.start.text)
                    #print(e.relation.name,e.start.text)
    except:
        pass
    return hiponimos

def jaro_distance(s1, s2,sinT,sinH,HipT,hipH):
    #print(s1, s2,sinT,sinH,HipT,hipH)
    bandera=True

    # Length of two strings
    len1 = len(s1)
    len2 = len(s2)

    # If the listas de tokens are equal 
    if len1==len2:
        for i in range(len1):
            if s1[i]!=s2[i]:
                bandera=False
                break
        if (bandera):
            return 1.0,1.0; 
 
    if (len1 == 0 or len2 == 0) :
        return 0.0,0.0; 
 
    # Maximum distance upto which matching 
    # is allowed 
    max_dist = (max(len(s1), len(s2)) // 2 )-1 ; 
 
    # Count of matches 
    match = 0; 
 
    # Hash for matches 
    hash_s1 = [0] * len(s1)
    hash_s2 = [0] * len(s2)
 
    # Traverse through the first string 
    for i in range(len1):
 
        # Check if there is any matches
        for j in range(max(0, i - max_dist), 
                       min(len2, i + max_dist + 1)):
            #print(s1[i],s2[j])
            # If there is a match or is contain in a bag of sinomys of tk
            if ((s1[i] == s2[j] or s1[i] in sinH[j] or s2[j] in sinT[i]) and hash_s2[j] == 0) : 
                print(s1[i],s2[j],"sinonimos")
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s1[i] in hipH[j] or len((sinT[i]).intersection(hipH[j]))>0) and hash_s2[j] == 0):
                print("hiponimos",s2[j],s1[i])#,(sinT[i]).intersection(hipH[j]))
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            elif ((s2[j] in HipT[i] or len((sinH[j]).intersection(HipT[i]))>0) and hash_s2[j] == 0): 
                print("hiperonimos sobre sinonimos",s2[j],s1[i])
                hash_s1[i] += 1; 
                hash_s2[j] += 1; 
                match += 1; 
                break
            # elif len((hipH[j]).intersection(HipT[i]))>0 and hash_s2[j] == 0: 
            #     print("hiperonimos3",s2[j],s1[i],(hipH[j]).intersection(HipT[i]))
            #     hash_s1[i] += 1; 
            #     hash_s2[j] += 1; 
            #     match += 1; 
            #     break
            
    print(hash_s1)
    print(hash_s2)
    print(match)
    # If there is no match 
    if (match == 0) :
        return 0.0,0.0; 
 
    # Number of transpositions 
    t = 0; 
 
    point = 0; 
 
    # Count number of occurrences 
    # where two characters match but 
    # there is a third matched character 
    # in between the indices 
    for i in range(len1) : 
        if (hash_s1[i]) :
            # Find the next matched character 
            # in second string 
            while (hash_s2[point] == 0) :
                point += 1; 
 
            if (s1[i] != s2[point]) :
                point += 1
                t += 1
            else :
                point += 1    
    t /= 2; 
    #Return the Jaro Similarity 
    return match / len2,(( match / len2  + match / len1 +
            (match - t) / match ) / 3.0); 

def relacion_noentailment(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh):
            if wt == e.start.text and e.relation.name in ["distinct_from","antonym"]:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

def relacion_conceptual(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh,two_way=True):# se agrego los dos maneras 
            if wt==e.end.text and e.relation.name in relaciones_generales: #estas relaciones 
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
            elif e.relation.name in ["related_to","similar_to"]:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

def negacion(nlp,texto):
    b=1.0
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return 0,""
    doc = nlp(texto.lower())
    for token in doc:
        if(token.dep_=="neg"):
            return 1, token.head.lemma_
    return 0,""

def representacion_entidadesDavid(nlp,texto):
    dir_sust=dict()
    palabras=[]
    b=1.0
    if (type(texto)==type(b) or texto=="" or texto=="n/a" or texto=="nan"):
        return dir_sust,palabras
    pos=[]
    lemmas=[]
    tokens=[]
    tokenshead=[]
    tokenschild=[]
    entidades=[]
    doc =nlp(texto.lower())
    for token in doc:
        #print([child for child in token.children],token.text, token.lemma_, token.pos_,token.dep_,token.head.text,token.head.lemma_, token.head.pos_)
        if token.text == "nobody" or token.text == "one":
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ not in ["DET","ADP","AUX","ADV","ADJ","NUM","PRON"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if token.pos_ in ["VERB"]:
                        if child.pos_ not in ["NOUN","VERB","PRON"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                        else:
                            if token.lemma_ !="be":
                                entidades.append(("","<UKN>",token.lemma_,token.pos_))
                    elif token.pos_ in ["NOUN"]:
                        if child.pos_ not in ["VERB"]:
                            entidades.append((child.lemma_,child.pos_,token.lemma_,token.pos_))
                    else:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        elif token.pos_ in ["ADJ"]:
            if (len(list(token.children))>0):
                for child in token.children:
                    if child.pos_ in ["ADP","AUX"]:
                        entidades.append(("","<UKN>",token.lemma_,token.pos_))
            else:
                entidades.append(("","<UKN>",token.lemma_,token.pos_))
        pos.append(token.pos_)
        lemmas.append(token.lemma_)
        tokens.append(token.text)
        tokenshead.append(token.head.text)
        tokenschild.append([child for child in token.children])
    print(entidades)
    dir_entidades=dict()
    for e in entidades:
        print(e[2])
        if e[3] not in ['PUNCT','CCONJ']:
            if e[2] not in dir_entidades and e[2] not in ["not"]:
                if str(e[0]) in ["no"]:
                    dir_entidades[str(e[2])]=str(e[0])
                elif e[1] in ["<UNK>","DET","ADP",'CCONJ','PRON']:
                    dir_entidades[e[2]]=""
                else:
                    # if e[1] in ["NOUN"]:
                    #     dir_entidades[str(e[2])+" "+str(e[0])]=""
                    if e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=""
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=""
                    else:
                        if e[1] not in ["PRON","PUNCT"]:# segundo agregue
                            dir_entidades[e[2]]=str(e[0])
            else:
                if e[2] not in ["not"]: #checar
                    if str(e[0]) in ["no"]:
                        dir_entidades[str(e[2])]=str(dir_entidades[e[2]])+","+str(e[0])
                    # elif e[1] in ["NOUN"]:
                    #         dir_entidades[str(e[2])+" "+str(e[0])]=""
                    elif e[1] in ["NOUN"]:
                        if e[0] not in dir_entidades:
                            dir_entidades[str(e[0])]=""
                        if e[2] not in dir_entidades:
                            dir_entidades[str(e[2])]=""
                    elif e[1] not in ["<UNK>","DET","ADP",'CCONJ','PRON',"PUNCT"]:
                        if str(dir_entidades[str(e[2])])=="":
                            dir_entidades[str(e[2])]=str(e[0])
                        else:
                            dir_entidades[str(e[2])]=str(dir_entidades[e[2]])+","+str(e[0])+","
    # print(pos)
    # print(lemmas)
    # print(tokens)
    # print(tokenshead)
    # print(tokenschild)
    return dir_entidades,list(dir_entidades.keys())

def relacion_entailmentF(token_h,token_t):
    if token_t in df_diccionario_generales and token_h in df_diccionario_generales:
        sinH=df_diccionario_generales[token_h]["synonym"]
        temp_set=set()
        # se usan las mismas relacines pero en el otro sentido para la hiponimia
        for p_g in relaciones_generales1:
            temp_set=temp_set.union(df_diccionario_especificas[token_h][p_g])
        hipH=temp_set
        if token_t in sinH:
            print("es sinonimo",token_t)
            return True
        elif token_t in hipH:
            print("es hiponimo",token_t)
            return True
    return False
    # for s_h in sinH:
    #     if s_h in :
    #         print("es sinonimo")
    #         cont1=True
    #         key=s_h
    # for s_h in hipH:
    #     if s_h in r_t:
    #         print("es sinonimo")
    #         cont1=True
    #         key=s_h
def relacion_noentailmentF(wt,wh):
    if wt in df_diccionario_generales and wh in df_diccionario_generales:
        antonimos_t=df_diccionario_generales[wt]["antonym"]
        cohiponimos_t=df_diccionario_generales[wt]["distinct_from"]
        antonimos_h=df_diccionario_generales[wh]["antonym"]
        cohiponimos_h=df_diccionario_generales[wh]["distinct_from"]
     
        if(wt in antonimos_h or wh in antonimos_t):
            print(wt,wh,"son antonimos: ")
            return True
        elif (wt in cohiponimos_h or wh in cohiponimos_t):
            print(wt,wh,"son cohiponimos: ")
            return True
    return False

def eliminacion_espacios(lista):
    eliminar_espacios=lista.count("")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("")
    eliminar_espacios=lista.count("be")
    if eliminar_espacios>0:
        for espacios in range(eliminar_espacios):
            lista.remove("be")
    return lista

def relacion_entailment(wt,wh):
    try:
        concepts_wt = Label.get(text=wt, language='en').concepts
        concepts_wh = Label.get(text=wh, language='en').concepts
        for e in edges_between(concepts_wt, concepts_wh):
            if wt == e.start.text and e.relation.name in relaciones_generales:
                print(e.start.text, "-", e.end.text, "|", e.relation.name,e)
                return True
    except:
        pass
    return False

nlp = spacy.load("en_core_web_md") # modelo de nlp

#ut.load_vectors_in_lang(nlp,"../OPENAI/data/glove.840B.300d.txt") # carga de vectores en nlp.wv
ut.load_vectors_in_lang(nlp,"./data/numberbatch-en-17.04b.txt") # carga de vectores en nlp.wv

#prueba=pd.read_csv("data/DEV/pruebaDEV.csv")
prueba=pd.read_csv("../OPENAI/data/"+sys.argv[1])

textos = prueba["sentence1"].to_list()       # almacenamiento en listas
hipotesis = prueba["sentence2"].to_list()
clases = prueba["gold_label"].to_list()


# lista de listas para dataframe
new_data = {'relation':[], 'no_matcheadas':[], 'contradiction':[],
            'distancias' : [], 'entropia_total' : [],'entropias' : [],'jaccard':[],'simBoW':[],
            'mutinf' : [], 'mearts' : [], 'max_info' : [], 'sumas' : [],'semantics':[], 
            'nlp_semantics':[],'mutinf_t' : [], 'mearts_t' : [], 'max_info_t' : [], 'sumas_t' : [],
            'entail':[],'contra':[],'neutral':[],'no_match':[],'rel_conceptuales':[],
            'list_comp' : [], 'diferencias' :[], 'list_incomp':[], 'entropia_relaciones':[],
            'list_M' : [], 'list_m' : [], 'list_T' : [], 'Jaro-Winkler_rit':[], 'KL_divergence':[],
            'negT' : [],  'negH' : [], 'overlap_ent':[],'clases' : []}


# cargar los conjuntos de sinonimos, hyperonimos e hyponimos
diccionario_sinonimos=dict()
diccionario_hiperonimos=dict()
diccionario_hyponimos=dict()
#diccionario_antonimos=dict()

df_temp=pd.read_pickle("data/Synonyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_sinonimos[strings['word']]=strings['Synonym']

df_temp=pd.read_pickle("data/Hyperonyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_hiperonimos[strings['word']]=strings['Hyperonym']

df_temp=pd.read_pickle("data/Hyponyms.pickle")
for index,strings in df_temp.iterrows():
    diccionario_hyponimos[strings['word']]=strings['Hyponym']

#cargar relaciones para trabajar en estas
df_diccionario = pd.read_pickle("data/Relaciones_generales.pickle")
df_diccionario_generales = df_diccionario.to_dict()

df_diccionario = pd.read_pickle("data/Relaciones_especificas.pickle")
df_diccionario_especificas = df_diccionario.to_dict()

inicio = time.time()
for i in range(len(textos)):
#for i in range(4):
    print(i)
    print(textos[i])
    
    ###############################################
    r_t,t_clean_m=representacion_entidadesDavid(nlp,textos[i])
    print(r_t,t_clean_m)
    print(hipotesis[i])
    r_h,h_clean_m = representacion_entidadesDavid(nlp,hipotesis[i])
    print(r_h,h_clean_m)
    
    #usamos las entidades para revisar y crear matriz de alineamiento
    if(type(hipotesis[i])!=type(1.0)):
        l_t=ut.get_lemmasR_(textos[i],nlp)
        l_h=ut.get_lemmasR_(hipotesis[i],nlp)
        s1=l_t
        s2=l_h
        #BoW
        v_vocabulario=set(s1).union(set(s2))
        t_bow=[]
        h_bow=[]
        for e in v_vocabulario:
            t_bow.append(s1.count(e))
            h_bow.append(s2.count(e))
        #print(t_bow)
        #print(h_bow)
        resultBoW = 1 - spatial.distance.cosine(t_bow, h_bow)
    else:
        s1=[]
        s2=[]
        l_t=[]
        l_h=[]
        resultBoW=1
    print(s1)
    print(s2)
    
    sinT=[]
    antT=[]
    HipT=[]
    sinH=[]
    antH=[]
    HipH=[]
    hipH=[]
    
    # # encontrar bolsa de sinonimos de cada token
    for t in s1:
        if t in df_diccionario_generales:
            sinT.append(df_diccionario_generales[t]["synonym"])
            temp_set=set()
            for p_g in relaciones_generales1:
                temp_set=temp_set.union(df_diccionario_generales[t][p_g])
            HipT.append(temp_set)
        else:
            sinT.append(set(t))
            HipT.append(set(t))
    for h in s2:
        if h in df_diccionario_generales:
            sinH.append(df_diccionario_generales[h]["synonym"])
            temp_set=set()
            # se usan las mismas relacines pero en el otro sentido para la hiponimia
            for p_g in relaciones_generales1:
                temp_set=temp_set.union(df_diccionario_especificas[h][p_g])
            hipH.append(temp_set)
        else:
            sinH.append(set(h))
            hipH.append(set(h))
    print(t_clean_m)
    print(h_clean_m)
    #print(s1, s2,sinT,sinH,HipT,hipH)
    tp1,tp2=jaro_distance(s1, s2,sinT,sinH,HipT,hipH)
    new_data['Jaro-Winkler_rit'].append(tp2)
    
    # proceso de eliminación de entidades y overlap las cosas que están compartiendo    
    lista_entidades_no_match=[]
    lista_entidades_distintas=[]
    lista_entidades_contenidas=[]
    for clave in r_h.keys():
        #print("hipotesis",clave)
        if clave in r_t:
            print("si esta",clave)
            t_atributos = eliminacion_espacios(r_t[clave].split(","))
            h_atributos = eliminacion_espacios(r_h[clave].split(","))
            print("atributos de T",t_atributos)
            print("atributos de H",h_atributos)
            matches=0
            if "no" in h_atributos or "no" in t_atributos or "not" in t_atributos:
                print("CONTRADICTION")
                lista_entidades_distintas.append(clave)
            elif len(h_atributos)>0:
                for h_a in h_atributos:
                    if h_a in t_atributos:
                        print("si esta",h_a)
                        matches+=1
                    else:
                        print("busqueda",h_a)
                        att_found=False
                        for attT in t_atributos:
                            if relacion_entailmentF(h_a,attT):
                                att_found=True
                                print("se encontró en relacion entailment",attT)
                                break
                        if att_found:
                            matches+=1
                if matches==len(h_atributos):
                    print("ENTAILMENT")
                    lista_entidades_contenidas.append(clave)
                else:
                    print("Entidad tiene más atributos - NEUTRAL")
                    lista_entidades_no_match.append(clave)
            else:
                print("ENTAILMENT")
                lista_entidades_contenidas.append(clave)
        else:
            for entT in list(r_t.keys()):
                if relacion_entailmentF(clave,entT):
                    matches=0
                    t_atributos = eliminacion_espacios(r_t[entT].split(","))
                    h_atributos = eliminacion_espacios(r_h[clave].split(","))
                    print("atributos de T",t_atributos)
                    print("atributos de H",h_atributos)
                    if "no" in h_atributos or "no" in t_atributos or "not" in t_atributos:
                        print("CONTRADICTION")
                        lista_entidades_distintas.append(clave)
                        break
                    elif len(h_atributos)>0:
                        for h_a in h_atributos:
                            if h_a in t_atributos:
                                print("si esta",h_a)
                                matches+=1
                            else:
                                print("busqueda",h_a)
                                att_found=False
                                for attT in t_atributos:
                                    if relacion_entailmentF(h_a,attT):
                                        att_found=True
                                        print("se encontro en relacion entailment",attT)
                                        break
                                if att_found:
                                    matches+=1
                        if matches==len(h_atributos):
                            print("ENTAILMENT")
                            lista_entidades_contenidas.append(clave)
                            break
                        else:
                            print("Entidad tiene más atributos - NEUTRAL")
                            lista_entidades_no_match.append(clave)
                            break
                    else:
                        print("ENTAILMENT")
                        lista_entidades_contenidas.append(clave)
                        break
                elif(relacion_noentailmentF(clave,entT)):
                    print("CONTRADICTION")
                    lista_entidades_distintas.append(clave)
                else:
                    print("no esta",clave)
                    lista_entidades_no_match.append(clave)
                    break
    print("----------------------------------------------")
    print("Contenidas",lista_entidades_contenidas)
    print("Faltantes",lista_entidades_no_match)
    print("Contradiccion",lista_entidades_distintas)
    if len(lista_entidades_distintas)>0:
        new_data["relation"].append(-1)
        #new_data["relation"].append("CONTRADICTION")
    elif len(lista_entidades_no_match)>0:
        new_data["relation"].append(2)
        #new_data["relation"].append("NEUTRAL")
    else:
        #new_data["relation"].append("ENTAILMENT")
        new_data["relation"].append(1)
    
    if len(h_clean_m)==0:
        new_data["overlap_ent"].append(0)
        new_data["no_matcheadas"].append(0)
        new_data["contradiction"].append(0)
    else:
        new_data["overlap_ent"].append(len(lista_entidades_contenidas)/len(h_clean_m))
        new_data["no_matcheadas"].append(len(lista_entidades_no_match)/len(h_clean_m))
        new_data["contradiction"].append(len(lista_entidades_distintas)/len(h_clean_m))
    new_data['simBoW'].append(resultBoW)
    
    ###############################################################################
    
    neg_t,negadat=negacion(nlp,textos[i])
    new_data['negT'].append(neg_t)
    print(hipotesis[i])
    neg_h,negadah=negacion(nlp,hipotesis[i])
    new_data['negH'].append(neg_h)
    # for clave in r_h.keys():
    #     print("hipotesis",clave,r_h[clave])
    if len(set(h_clean_m))!=0 and len(set(t_clean_m))!=0:
        new_data['jaccard'].append(len(set(t_clean_m).intersection(set(h_clean_m)))/len(set(h_clean_m)))
    else:
        new_data['jaccard'].append(0)


    
    t_lem=list(set(ut.get_lemmas_(textos[i],nlp)))
    h_lem=list(set(ut.get_lemmas_(hipotesis[i],nlp)))
    t_vectors=ut.get_matrix_rep2(t_lem, nlp, normed=False)
    h_vectors=ut.get_matrix_rep2(h_lem, nlp, normed=False)
    t_vectors_n=ut.get_matrix_rep2(t_lem, nlp, normed=True)
    h_vectors_n=ut.get_matrix_rep2(h_lem, nlp, normed=True)
    
    # # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
    ma=np.dot(t_vectors_n,h_vectors_n.T)
    #print(t_clean,h_clean)
    #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
    m_earth,m_mi=wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_lem,h_lem)
    ma=pd.DataFrame(ma,index=t_lem,columns=h_lem)
    #print(ma)
    new_data['max_info_t'].append(ma.max().sum()/(ma.shape[1]))#
    new_data['sumas_t'].append(ma.sum().sum()/((ma.shape[1]*(ma.shape[0]))))#
    new_data['mearts_t'].append(m_earth.min().sum()/(ma.shape[1]))# 
    new_data['mutinf_t'].append(m_mi.max().sum()/(ma.shape[1]))# 
    # # Calculamos la entropia inicial de la matriz de distancias coseno sobre tokens de T y H
    distX = ma.round(1).values.flatten()
    new_data['entropia_total'].append(entropia(distX)) 

    # ###### BORRADO DE COSAS QUE NO OCUPO, SOLO NOS QUEDAMOS CON INFORMACIÓN DE TIPOS DE PALABRA: NOUN, VERB, ADJ Y ADV
    # # TAMBIÉN OMITIMOS EL VERBO BE DEBIDO A QUE POR LO REGULAR SE UTILIZA COMO AUXILIAR Y ES UN VERBO COPULATIVO
    # # sirve para construir la llamada predicación nominal del sujeto de una oración: 
    # # #el sujeto se une con este verbo a un complemento obligatorio llamado atributo que por lo general determina 
    # # alguna propiedad, estado o equivalencia del mismo, por ejemplo: "Este plato es bueno". "Juan está casado".
    c_compatibilidad=0
    c_incompatibilidad=0
    c_rel_concep=0
    b_col=[0]

    new_data['list_T'].append(ma.shape[0])
    new_data['list_M'].append(ma.shape[1])
    print(ma)
    # val=ma.max().values
    # print("valores maximos",val.round(1))
    # print("entropias de valores maximos",entropia(val.round(1)))

    #procesamiento de cosas que son la misma entidad
    borrar=list(set(t_lem).intersection(set(h_lem)))
    ma = ma.drop(borrar,axis=1)
    m_earth = m_earth.drop(borrar,axis=1)
    m_mi = m_mi.drop(borrar,axis=1)
    b_col.extend(borrar)

    #Como son palabras iguales entonces se agregan como uno que significa una realcion de entailment
    #calculo de relaciones con entropia
    rel_entropia=[]
    for b_c in borrar:
        rel_entropia.append(1)

    # a = ma.idxmax().values
    # b = ma.columns
    top_k=3
    # # #PARA REVISAR SI EXISTEN RELACIONES DE SIMILITUD SEMÁNTICA A TRAVÉS DEL USO DE CONCEPNET
    print("proceso de obtención de generalidad")
    print(ma,ma.columns)
    for c_c in ma.columns:
        print("columna a checar",c_c)
        # filtrar el top 3 de los mejores similitud coseno para cada token de H vs tokens de T que sean mayores a 0
        # una vez que encontremos quien se sale del ciclo
        temp=ma[c_c].sort_values(ascending=False)
        ranks=list(temp[:top_k].index)
        valranks=list(temp[:top_k].values)
        for r_i in range(len(ranks)):
            borrar=[]
            print("acces",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)
                if(relacion_entailment(r_wt,r_wh)):
                    borrar.append(r_wh)
                    rel_entropia.append(1)
                    c_compatibilidad+=1
                    break
                else:
                    print("Proceso de conjuntos")
                    if r_wt in diccionario_sinonimos and r_wh in diccionario_sinonimos:
                        sin1=diccionario_sinonimos[r_wt]
                        sin2=diccionario_sinonimos[r_wh]
                    else:
                        sin1=set()
                        sin2=set()
                    if len(sin1.intersection(sin2))>0:
                        borrar.append(r_wh)
                        c_compatibilidad+=1
                        rel_entropia.append(1)
                        break
                    else:
                        Hip1=set()
                        for e in list(sin1):
                            if e in diccionario_hiperonimos:
                                Hip1=Hip1.union(diccionario_hiperonimos[e])
                            else:
                                b_H=bag_of_hyperonyms(e)
                                Hip1=Hip1.union(b_H)
                                diccionario_hiperonimos[e]=b_H
                        if len(Hip1.intersection(sin2))>0:
                            borrar.append(r_wh)
                            c_compatibilidad+=1
                            rel_entropia.append(1)
                            break
                        else:
                            hip2=set()
                            for e in list(sin2):
                                if e in diccionario_hyponimos:
                                    hip2=hip2.union(diccionario_hyponimos[e])
                                else:
                                    b_H=bag_of_hyponyms(e)
                                    hip2=hip2.union(b_H)
                                    diccionario_hyponimos[e]=b_H                                
                            if len(sin1.intersection(hip2))>0:   
                                borrar.append(r_wh)
                                c_compatibilidad+=1
                                rel_entropia.append(1)
                                break
        ma = ma.drop(borrar,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)

    # proceso para saber si hay contradiction en los que faltan
    print("proceso de obtención de contradiction")
    print(ma,ma.columns)
    for c_c in ma.columns:
        print("columna a checar para contradicción",c_c)
        # filtrar el top 3 de los mejores similitud coseno para cada token de H vs tokens de T que sean mayores a 0
        # una vez que encontremos quien se sale del ciclo
        temp=ma[c_c].sort_values(ascending=False)
        ranks=list(temp[:top_k].index)
        valranks=list(temp[:top_k].values)
        for r_i in range(len(ranks)):
            borrar=[]
            print("acces",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)
                if(relacion_noentailment(r_wt,r_wh)):
                    c_incompatibilidad+=1
                    rel_entropia.append(0)
                    #borrar.append(r_wh)
                    break
                #else:
                    # sin1=diccionario_sinonimos[r_wt]
                    # sin2=diccionario_sinonimos[r_wh]
                    # Hip1=set()
                    # for e in list(sin1):
                    #     if e in diccionario_hiperonimos:
                    #         Hip1=Hip1.union(diccionario_hiperonimos[e])
                    #     else:
                    #         b_H=bag_of_hyperonyms(e)
                    #         Hip1=Hip1.union(b_H)
                    #         diccionario_hiperonimos[e]=b_H
                    # Hip2=set()
                    # for e in list(sin2):
                    #     if e in diccionario_hiperonimos:
                    #         Hip2=Hip2.union(diccionario_hiperonimos[e])
                    #     else:
                    #         b_H=bag_of_hyperonyms(e)
                    #         Hip2=Hip2.union(b_H)
                    #         diccionario_hiperonimos[e]=b_H
                    # Hip1=diccionario_hiperonimos[r_wt]
                    # Hip2=diccionario_hiperonimos[r_wh]
                    # if len(Hip1.intersection(Hip2))>0:
                    #     c_incompatibilidad+=1
                    #     rel_entropia.append(0)
                    #     borrar.append(r_wh)
                    #     break

        ma = ma.drop(borrar,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)

    # proceso para checar las relaciones conceptuales que existen
    print("proceso de obtención de especificidad y conceptuales")
    print(ma,ma.columns)
    for c_c in ma.columns:
        print("columna a checar para especificidad",c_c)
        
        # filtrar el top 3 de los mejores similitud coseno para cada token de H vs tokens de T que sean mayores a 0
        # una vez que encontremos quien se sale del ciclo
        temp=ma[c_c].sort_values(ascending=False)
        ranks=list(temp[:top_k].index)
        valranks=list(temp[:top_k].values)
        for r_i in range(len(ranks)):
            borrar=[]
            print("accesar a checar conceptuales",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)                
                if (relacion_conceptual(r_wt,r_wh)):
                    rel_entropia.append(2)
                    c_rel_concep+=1
                    borrar.append(r_wh)
                    break

        ma = ma.drop(borrar,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)
    b_index=[0]
    
    # proceso para checar las no relaciones que existen
    print("proceso de obtención de no relaciones")
    print(ma,ma.columns)
    for c_c in ma.columns:
        rel_entropia.append(3)

    print("rel",np.array(rel_entropia).round(1))
    print("entropia final",entropia(np.array(rel_entropia).round(1)))

    new_data['entail'].append(rel_entropia.count(1)/len(rel_entropia))
    new_data['contra'].append(rel_entropia.count(0)/len(rel_entropia))
    new_data['neutral'].append(rel_entropia.count(2)/len(rel_entropia))
    new_data['no_match'].append(rel_entropia.count(3)/len(rel_entropia))
    new_data['entropia_relaciones'].append(entropia(np.array(rel_entropia).round(1)))

    #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA DE CARACTERÍSTICAS
    m_distancia = obtener_distancia(t_vectors,h_vectors,t_lem,h_lem,b_col,b_index)
    m_earth=m_earth*m_distancia

    if ma.shape[1]==0:
        new_data['entropias'].append(0)
        new_data['KL_divergence'].append(0)
        new_data['max_info'].append(0)
        new_data['sumas'].append(0)
        new_data['mearts'].append(0)
        new_data['mutinf'].append(0)
        new_data['diferencias'].append(0)
        new_data['distancias'].append(0)
        new_data['semantics'].append(1)
        if c_incompatibilidad>0:
            new_data['nlp_semantics'].append(0)
        elif c_rel_concep>0:
            new_data['nlp_semantics'].append(0.5)
        else:
            new_data['nlp_semantics'].append(1)
        #new_data['nlp_semantics'].append(1)
        #new_data['semantics'].append(1)
    else:
        distY = ma.round(1).values.flatten()
        new_data['entropias'].append(entropia(distY))
        new_data['KL_divergence'].append(kullback_leibler(distX,distY))
        new_data['max_info'].append(ma.max().sum()/(ma.shape[1]))#
        new_data['sumas'].append(ma.sum().sum()/((ma.shape[1]*(ma.shape[0]))))#
        new_data['mearts'].append(m_earth.min().sum()/(ma.shape[1]))# 
        new_data['mutinf'].append(m_mi.max().sum()/(ma.shape[1]))# 
        new_data['diferencias'].append(len(ma.columns)/len(ma.index))
        new_data['distancias'].append(m_distancia.min().sum()/(ma.shape[1]))
        doc1 = nlp(" ".join(ma.idxmax().values))
        doc2 = nlp(" ".join(ma.columns.values))
        new_data['semantics'].append(float(doc1.similarity(doc2)))
        if c_incompatibilidad>0:
            new_data['nlp_semantics'].append(1-float(doc1.similarity(doc2)))
        else:
            new_data['nlp_semantics'].append(float(doc1.similarity(doc2)))

    new_data['list_comp'].append(c_compatibilidad)
    new_data['list_incomp'].append(c_incompatibilidad)
    new_data['rel_conceptuales'].append(c_rel_concep)
    new_data['list_m'].append(ma.shape[1])
    new_data['clases'].append(clases[i])
    print(ma)

df_resultados = pd.DataFrame(new_data)
df_resultados.to_pickle("salida/nuevo2/"+sys.argv[1]+"_.pickle")
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")