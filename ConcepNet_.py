import pandas as pd
import conceptnet_lite
conceptnet_lite.connect("../OPENAI/data/conceptnet.db")
from conceptnet_lite import Label, edges_for, edges_between

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