import pandas as pd
import numpy as np

import utils as ut # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import ConcepNet_ as cn # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import utilidades as utils # esta librería tiene funciones para poder obtener un procesamiento del <T,H>
import processTxt as ptxt
import spacy
import time
import sys
from math import floor
from scipy import spatial

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
            'negT' : [],  'negH' : [], 'overlap_ent':[], 'relacionesEncontradas':[],'Texto':[],'Hipotesis':[],
            'TextoL':[],'HipotesisL':[],'pInflexion':[],'pInflexionV':[],'entSimilitud':[],'dicEntT':[],'dicEntH':[],
            'H_grupo1':[],'h_k_grupo1':[],'H_grupo2':[],'h_k_grupo2':[],'H_grupo3':[],'h_k_grupo3':[],
            'clases' : []}

inicio = time.time()
for i in range(len(textos)):
#for i in range(4):
    print(i)
    print(textos[i])
    
    ###############################################
    r_t,t_clean_m=ptxt.representacion_entidadesDavid(nlp,textos[i])
    print(r_t,t_clean_m)
    print(hipotesis[i])
    r_h,h_clean_m = ptxt.representacion_entidadesDavid(nlp,hipotesis[i])
    print(r_h,h_clean_m)
    
    new_data['dicEntT'].append(r_t)
    new_data['dicEntH'].append(r_h)
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
        if t in cn.df_diccionario_generales:
            sinT.append(cn.df_diccionario_generales[t]["synonym"])
            temp_set=set()
            for p_g in cn.relaciones_generales1:
                temp_set=temp_set.union(cn.df_diccionario_generales[t][p_g])
            HipT.append(temp_set)
        else:
            sinT.append(set(t))
            HipT.append(set(t))
    for h in s2:
        if h in cn.df_diccionario_generales:
            sinH.append(cn.df_diccionario_generales[h]["synonym"])
            temp_set=set()
            # se usan las mismas relacines pero en el otro sentido para la hiponimia
            for p_g in cn.relaciones_generales1:
                temp_set=temp_set.union(cn.df_diccionario_especificas[h][p_g])
            hipH.append(temp_set)
        else:
            sinH.append(set(h))
            hipH.append(set(h))
    print(t_clean_m)
    print(h_clean_m)
    #print(s1, s2,sinT,sinH,HipT,hipH)
    tp1,tp2=cn.jaro_distance(s1, s2,sinT,sinH,HipT,hipH)
    new_data['Jaro-Winkler_rit'].append(tp2)
    
    # proceso de eliminación de entidades y overlap las cosas que están compartiendo    
    lista_entidades_no_match=[]
    lista_entidades_distintas=[]
    lista_entidades_contenidas=[]
    for clave in r_h.keys():
        #print("hipotesis",clave)
        if clave in r_t:
            print("si esta",clave)
            t_atributos = ptxt.eliminacion_espacios(r_t[clave].split(","))
            h_atributos = ptxt.eliminacion_espacios(r_h[clave].split(","))
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
                            if cn.relacion_entailmentF(h_a,attT):
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
                if cn.relacion_entailmentF(clave,entT):
                    matches=0
                    t_atributos = ptxt.eliminacion_espacios(r_t[entT].split(","))
                    h_atributos = ptxt.eliminacion_espacios(r_h[clave].split(","))
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
                                    if cn.relacion_entailmentF(h_a,attT):
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
                elif(cn.relacion_noentailmentF(clave,entT)):
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
    
    neg_t,negadat=ptxt.negacion(nlp,textos[i])
    new_data['negT'].append(neg_t)
    print(hipotesis[i])
    neg_h,negadah=ptxt.negacion(nlp,hipotesis[i])
    new_data['negH'].append(neg_h)
    # for clave in r_h.keys():
    #     print("hipotesis",clave,r_h[clave])
    if len(set(h_clean_m))!=0 and len(set(t_clean_m))!=0:
        new_data['jaccard'].append(len(set(t_clean_m).intersection(set(h_clean_m)))/len(set(h_clean_m)))
    else:
        new_data['jaccard'].append(0)

    #t_lem=list(set(ut.get_lemmas_(textos[i],nlp)))
    #h_lem=list(set(ut.get_lemmas_(hipotesis[i],nlp)))

    t_lem_temp=ut.get_lemmas_(textos[i],nlp)
    h_lem_temp=ut.get_lemmas_(hipotesis[i],nlp)
    t_lem=[]
    h_lem=[]
    for a in t_lem_temp:
        if a not in t_lem:
            t_lem.append(a)
    for a in h_lem_temp:
        if a not in h_lem:
            h_lem.append(a)


    t_vectors=ut.get_matrix_rep2(t_lem, nlp, normed=False)
    h_vectors=ut.get_matrix_rep2(h_lem, nlp, normed=False)
    t_vectors_n=ut.get_matrix_rep2(t_lem, nlp, normed=True)
    h_vectors_n=ut.get_matrix_rep2(h_lem, nlp, normed=True)
    
    redondeo=2
    ma_n=np.dot(t_vectors_n,h_vectors_n.T)
    ma_n = np.clip(ma_n, 0, 1).round(redondeo)
    
    ma=pd.DataFrame(ma_n,index=t_lem,columns=h_lem)
    
    # vamos a crear una copia para aplicarles diferentes procesos de eliminacion dependiendo el grupo
    # una para cada grupo
    ma_original = ma.copy()
    ma_generalidad = ma.copy()
    ma_contradiction = ma.copy()
    ma_especificidad = ma.copy()
    


    simC1 = len(np.where(ma_original.values >= 0.75)[0])
    simC2 = len(np.where((ma_original.values >= 0.5) & (ma_original.values < 0.75))[0])
    simC3 = len(np.where(ma_original.values < 0.5)[0])

    distXSim=[]

    distX = ma.round(1).values.flatten()
    for m_sim in range(simC1):
        distXSim.append(1)
    for m_sim in range(simC2):
        distXSim.append(2)
    for m_sim in range(simC3):
        distXSim.append(0)

    print("rel nueva categoria",np.array(distXSim).round(1))

    new_data['entSimilitud'].append(utils.entropia(np.array(distXSim).round(1)))    


    print(ma)
    print("Tokens de H: ",ma.columns)
    print("Tokens de T: ",ma.index)
    H_T_H = ut.entropia(ma.values.flatten())
    print("Entropia de ma: ",H_T_H)
    print(ma_n.shape)
    hs=np.array([])
    ent_anterior=0
    cambio=0
    pVectores=[]
    for j in range(ma_n.shape[1]):
        print(ma.columns.values[j])
        hk=ma_n[:,j].flatten()
        print("Valores de",j, "dado T: ",hk)
        H_h_T = ut.entropia(ma_n[:,j].flatten())
        print("Entropia de",j, "dado T: ",H_h_T)
        print("Diferencia H(T,H)- H(h,T)",H_T_H-H_h_T)
        hs = np.append(hs,hk)
        ent_hs= ut.entropia(hs)
        pVectores.append(ent_hs)
        if (ent_anterior>ent_hs):
            cambio+=1
        print(ent_anterior,"Entropia de ",hs, " incrementando h",j," :",ent_hs)
        ent_anterior=ent_hs
        if j==0:
            h_1 = ent_hs
    new_data['pInflexion'].append(cambio)
    new_data['pInflexionV'].append(H_T_H - h_1)

    # # Obtencion de matriz de alineamiento, matriz de move earth y mutual information
    # ma=np.dot(t_vectors_n,h_vectors_n.T)
    # print(ma)

    #print(t_clean,h_clean)
    #print(len(t_vectors_n),len(h_vectors_n),len(t_clean),len(h_clean))
    m_earth,m_mi=utils.wasserstein_mutual_inf(t_vectors_n,h_vectors_n,t_lem,h_lem)

    new_data['TextoL'].append(ut.get_words_rep(textos[i],nlp))
    new_data['HipotesisL'].append(ut.get_words_rep(hipotesis[i],nlp))

    ma=pd.DataFrame(ma,index=t_lem,columns=h_lem)
    #print(ma)
    new_data['max_info_t'].append(ma.max().sum()/(ma.shape[1]))#
    new_data['sumas_t'].append(ma.sum().sum()/((ma.shape[1]*(ma.shape[0]))))#
    new_data['mearts_t'].append(m_earth.min().sum()/(ma.shape[1]))# 
    new_data['mutinf_t'].append(m_mi.max().sum()/(ma.shape[1]))# 
    # # Calculamos la entropia inicial de la matriz de distancias coseno sobre tokens de T y H
    distX = ma.round(1).values.flatten()
    new_data['entropia_total'].append(utils.entropia(distX)) 

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
    borrar=[]
    #borrar=list(set(t_lem).intersection(set(h_lem)))
    ma = ma.drop(borrar,axis=1)
    m_earth = m_earth.drop(borrar,axis=1)
    m_mi = m_mi.drop(borrar,axis=1)
    b_col.extend(borrar)

    #Como son palabras iguales entonces se agregan como uno que significa una realcion de entailment
    #calculo de relaciones con entropia
    rel_entropia=[]
    for b_c in borrar:
        rel_entropia.append(1)

    tuplas_relaciones=""

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
            borrar_n=[]
            print("acces",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)
                if(cn.relacion_entailment(r_wt,r_wh)):
                    borrar.append(r_wh)
                    borrar_n.append(r_wh)
                    rel_entropia.append(1)
                    c_compatibilidad+=1
                    tuplas_relaciones+=" Generalidad entailment: "+r_wt+"-"+r_wh+" | "
                    break
                else:
                    print("Proceso de conjuntos")
                    if r_wt in cn.diccionario_sinonimos and r_wh in cn.diccionario_sinonimos:
                        sin1=cn.diccionario_sinonimos[r_wt]
                        sin2=cn.diccionario_sinonimos[r_wh]
                    else:
                        sin1=set()
                        sin2=set()
                    if len(sin1.intersection(sin2))>0:
                        borrar.append(r_wh)
                        borrar_n.append(r_wh)
                        c_compatibilidad+=1
                        rel_entropia.append(1)
                        tuplas_relaciones+=" Generalidad, Sinonimos: "+r_wt+"-"+r_wh+" | "
                        break
                    else:
                        Hip1=set()
                        for e in list(sin1):
                            if e in cn.diccionario_hiperonimos:
                                Hip1=Hip1.union(cn.diccionario_hiperonimos[e])
                            else:
                                b_H=cn.bag_of_hyperonyms(e)
                                Hip1=Hip1.union(b_H)
                                cn.diccionario_hiperonimos[e]=b_H
                        if len(Hip1.intersection(sin2))>0:
                            borrar.append(r_wh)
                            borrar_n.append(r_wh)
                            c_compatibilidad+=1
                            rel_entropia.append(1)
                            tuplas_relaciones+=" Generalidad H, H y S: "+r_wt+"-"+r_wh+" | "
                            break
                        else:
                            hip2=set()
                            for e in list(sin2):
                                if e in cn.diccionario_hyponimos:
                                    hip2=hip2.union(cn.diccionario_hyponimos[e])
                                else:
                                    b_H=cn.bag_of_hyponyms(e)
                                    hip2=hip2.union(b_H)
                                    cn.diccionario_hyponimos[e]=b_H                                
                            if len(sin1.intersection(hip2))>0:   
                                borrar.append(r_wh)
                                borrar_n.append(r_wh)
                                c_compatibilidad+=1
                                rel_entropia.append(1)
                                tuplas_relaciones+=" Generalidad, S e h: "+r_wt+"-"+r_wh+" | "
                                break
        ma = ma.drop(borrar,axis=1)
        ma_generalidad = ma_generalidad.drop(borrar_n,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)

    new_data['H_grupo1'].append(utils.entropia(ma_generalidad.values.flatten()))     
    new_data['h_k_grupo1'].append(utils.entropia(ma_original[borrar_n].values.flatten()))     
        
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
            borrar_n=[]
            print("acces",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)
                if(cn.relacion_noentailment(r_wt,r_wh)):
                    c_incompatibilidad+=1
                    rel_entropia.append(0)
                    borrar.append(r_wh)
                    borrar_n.append(r_wh)
                    tuplas_relaciones+=" Contradiction: "+r_wt+"-"+r_wh+" | "
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
        ma_contradiction = ma_contradiction.drop(borrar_n,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)

    new_data['H_grupo2'].append(utils.entropia(ma_contradiction.values.flatten()))     
    new_data['h_k_grupo2'].append(utils.entropia(ma_original[borrar_n].values.flatten()))     
    
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
            borrar_n=[]
            print("accesar a checar conceptuales",c_c,ranks[r_i],valranks[r_i])
            if valranks[r_i]>0:
                r_wt=str(ranks[r_i])
                r_wh=str(c_c)                
                if (cn.relacion_conceptual(r_wt,r_wh)):
                    rel_entropia.append(2)
                    c_rel_concep+=1
                    tuplas_relaciones+=" Especificidad: "+r_wt+"-"+r_wh+" | "
                    borrar.append(r_wh)
                    borrar_n.append(r_wh)
                    break

        ma = ma.drop(borrar,axis=1)
        ma_especificidad = ma_especificidad.drop(borrar_n,axis=1)
        m_earth = m_earth.drop(borrar,axis=1)
        m_mi = m_mi.drop(borrar,axis=1)
        n_columns = ma.shape[1]
        b_col.extend(borrar)
    b_index=[0]
    
    
    new_data['H_grupo3'].append(utils.entropia(ma_especificidad.values.flatten()))     
    new_data['h_k_grupo3'].append(utils.entropia(ma_original[borrar_n].values.flatten()))     
    
    
    # proceso para checar las no relaciones que existen
    print("proceso de obtención de no relaciones")
    print(ma,ma.columns)
    for c_c in ma.columns:
        rel_entropia.append(3)

    print("rel",np.array(rel_entropia).round(1))
    print("entropia final",utils.entropia(np.array(rel_entropia).round(1)))

    new_data['entail'].append(rel_entropia.count(1)/len(rel_entropia))
    new_data['contra'].append(rel_entropia.count(0)/len(rel_entropia))
    new_data['neutral'].append(rel_entropia.count(2)/len(rel_entropia))
    new_data['no_match'].append(0)#rel_entropia.count(3)/len(rel_entropia))
    new_data['entropia_relaciones'].append(utils.entropia(np.array(rel_entropia).round(1)))

    #   ALMACENAMIENTO DE TODA LA INFORMACIÓN PROCESADA DE CARACTERÍSTICAS
    m_distancia = utils.obtener_distancia(t_vectors,h_vectors,t_lem,h_lem,b_col,b_index)
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
        new_data['entropias'].append(utils.entropia(distY))
        new_data['KL_divergence'].append(utils.kullback_leibler(distX,distY))
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
    new_data['relacionesEncontradas'].append(tuplas_relaciones)
    new_data['Texto'].append(textos[i])
    new_data['Hipotesis'].append(hipotesis[i])
    new_data['clases'].append(clases[i])
    print(ma)

df_resultados = pd.DataFrame(new_data)
df_resultados.to_pickle("salida/nuevo3/"+sys.argv[1]+"_.pickle")
fin = time.time()
print("Tiempo que se llevo:",round(fin-inicio,2)," segundos")