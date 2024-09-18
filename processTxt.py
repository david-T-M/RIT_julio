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