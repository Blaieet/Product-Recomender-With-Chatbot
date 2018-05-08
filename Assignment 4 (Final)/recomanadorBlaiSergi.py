
# coding: utf-8

# # Recomanació basada en PageRank
# 
# L'algorisme PageRank, famós per l'ús en el cercador de Google, te moltes altres aplicacions. Un exemple d'ells és la recomanació d'items a usuaris basat en les similituds entre usuaris.
# 
# Recordeu el principi de funcionament bàsic de les recomanacions colaboratives basades en l'usuari, obtenim una puntuació per a un item i usuari, basat en les similituds d'aquest usuari amb la resta, i respectives puntuacions. Si ho formulem, podríem dir que tot es basa en:
# 
# $$\hat{r}_{u,i} = \frac{\sum_{v,v\neq u} sim(u,v) \cdot r_{v,i}}{\sum_{v,v\neq u} sim(u,v)}$$
# 
# Amb notació 
# 
# * $r_{u,i}$ ens indica la puntuació ($r$ating) de l'usuari $u$ a l'item $i$. 
# * El sumatori $\sum_{v,v\neq u}$ indica la suma per cada usuari $v$ que no sigui el propi $u$ del que estem intentant predir una puntuació
# * El barret a $\hat{r}$ denota que es tracta d'una predicció, el valor que estem intentant inferir a partir de les dades.
# 
# La funció $sim$ és la que ens indica quan semblants són dos usuaris entre sí. Tal i com heu vist a teoria, això es pot fer amb mètriques com la distància euclidea o la similitud de Pearson, d'entre moltes altres. En aquesta pràctica, però, veure'm com el vector obtingut a partir de calcular PageRank sobre una matriu concreta d'usuaris i items, també ens proporciona una mesura de similitud significativa.

# ## Bibliografia
# 
# Per aquells interesats, tota la informació relativa a PageRank està basada en la publicació
# 
# `Bryan, K., & Leise, T. (2006). The $25,000,000,000 eigenvector: The linear algebra behind Google. Siam Review, 48(3), 569-581.`
# 
# I els algorismes matemàtiques en el llibre:
# 
# `Applied numerical linear algebra, James W. Demmel`, capítol 4.4 

# # PageRank
# 
# Aquesta cel·la serveix com a una breu recapitulació de l'algorisme PageRank vist a teoria. Recordem que PageRank es basa en trobar l'importància de les pàgines en base a la reputació d'aquestes i del número de links entrants i sortints.
# 
# Supossa que tenim la següent estructura de pàgines:

# <img src="img/page1.png" />

# És a dir, si comptem el número de `in-link` (enllaços d'entrada) de cada pàgina $x$, obtindríem:
# $$x_1=3,x_2=2,x_3=1,x_4=3$$
# 
# Però també podem expressar-ho en funció de la pàgina de qui rebem el link, de forma que tinguem la importància en compte:
# 
# $$x_{entrada} = \dfrac{x_{sortida}}{|x_{sortida}|}$$
# 
# Per exemple, si ho apliquem a $x_1$, $x_1 = x_2 / 2 + x_3 / 2 + x_4 / 3$. De forma semblant, podem aplicar-ho a la resta:
# 
# $$x_1 = x_2 / 2 + x_3 / 2 + x_4 / 3$$
# $$x_2 = x_1 / 2 + x_4 / 3$$
# $$x_3 = x_4 / 3$$
# $$x_4 = x_1 / 2 + x_2 / 2 + x_3 / 2$$
# 
# Ara podríem resoldre aquest sistema per trobar quina és la importància de cada web, un vector $(s_1,s_2,s_3,s_4)$. Però, resoldre-ho no és trivial (no en casos on tenim millors de webs, és clar!). Per tal de poder avançar, el que se sol fer és resoldre-ho en forma matricial:
# 
# $$
# \begin{array}{ccc}
# & \text{Out-links} &\\
# G = &\begin{bmatrix}
#     0 & 1/2 & 1/2 & 1/3 \\
#     1/2 & 0 & 0 & 1/3 \\
#     0 & 0 & 0 & 1/3 \\
#     1/2 & 1/2 & 1/2 & 0
# \end{bmatrix} & \text{In-links}
# \end{array}
# $$
# 
# El sistema ara es converteix en una equació, arquetip fàcilment reconegut pels matemàtics:
# 
# $$x = Gx$$
# 
# Trobareu els detalls a la publicació si esteu interesats, realment el que estem intentant és trobar el vector propi que té per valor propi 1. És a dir, resoldre $\lambda x = Gx$ tal que $\lambda = 1$

# ## Trobant el vector propi
# 
# Un dels mètodes més eficients, tot i que no el que més, és el mètode de la potència. Aquest, permet trobar el vector propi que té el valor propi més alt (coses de les matemàtiques, es pot demostrar que $\lambda=1$ serà el més alt). El mètode diu així:
# 
# $
# i = 0\\
# \text{do}\\
# \hspace{2cm}y_{i+1} = Gx_i\\
# \hspace{2cm}x_{i+1} = y_{i+1} / ||y_{i+1}||_2\\
# \hspace{2cm}i = i + 1\\
# \text{until }||x_{i+1} - x_{i}|| < 10^{-6}
# $
# 
# On $x_0$ és un vector normalitzat amb suma 1, per exemple si tenim $n$ webs en total $x_0 = \textbf{1} / n$, on $\textbf{1}$ és un vector d'1s de tamany $n$.

# **Programa l'algorisme del mètode de la potència amb numpy, seguint el pseudocodi de dalt**

# In[1]:

import numpy as np

def power_method(G):
    """
    Donada una matriu d'adjecències, en calcula el PageRank.
    Mitjançant el mètode de la potència troba el vector propi
    de valor propi màxim (1)
    
    :param G: Matriu a calcular el PageRank
    :return: Vector d'importàncies del PageRank (vector
        propi amb valor propi més alt)
    """
    dimensioG = G.shape[0]
    x = np.ones(dimensioG)/dimensioG #Normalitzem
    while True:
        xSeguent = G.dot(x) #Dot: multiplicacio
        xSeguent = xSeguent/np.linalg.norm(xSeguent) #Diviim per la normalitzacio del seguent
        if np.linalg.norm(xSeguent-x) < pow(10,-6): #Es tan petit com voliem?
            return xSeguent
        x = xSeguent #Si no, torna a iterar
        
        
def solve_eig(G):
    """
    Calcula els vectors i valors propis de la matriu G
    mitjançant funcions de numpy. Funció de referència
    que us pot servir per comprovar que el vostre mètode
    power_method retorna el que toca.
    
    :param G: Matriu a calcular el PageRank
    :return: Vector d'importàncies del PageRank (vector
        propi amb valor propi més alt)
    """
    vals, vecs = np.linalg.eig(G)
    idxs = np.argsort(np.real(vals))
    return np.abs(vecs[:, idxs[-1]])


# In[2]:

if __name__ == '__main__':
    G1 = np.asarray((
        (0,     1/2.0, 1/2.0, 1/3.0), 
        (1/2.0, 0,     0,     1/3.0), 
        (0,     0,     0,     1/3.0), 
        (1/2.0, 1/2.0, 1/2.0, 0)
    ))
    x = power_method(G1)
    y = solve_eig(G1)
    
    print(np.round(G1, 2))
    print('Eigenvector', np.round(x, 2))
    print('Eigenvector', np.round(y, 2))
    print('Eigenvalues', np.round(np.linalg.eigvals(G1), 2))
    
#Podem veure que obtenim els mateixos resultats:


# ## Casos extrems
# 
# Evidentment, no tot és tant bonic com sembla... Crea la matriu de la següent configuració i prova que passa quan n'executes el mètode de la potència:
# 
# <img src="img/page2.png" />

# In[3]:

if __name__ == '__main__':
    G2 = np.asarray((
        (0,     1/2.0, 1/2.0, 0,   0), 
        (1/2.0, 0,     1/2.0, 0,   0), 
        (1/2.0, 1/2.0, 0,     0,   0), 
        (0,     0,     0,     0,   1.0),
        (0,     0,     0,     1.0, 0)
    ))
    x = power_method(G2)
    y = solve_eig(G2)
    
    print(np.round(G2, 2))
    print('Eigenvector', np.round(x, 2))
    print('Eigenvector', np.round(y, 2))
    print('Eigenvalues', np.round(np.linalg.eigvals(G2), 2))


# I encara en trobem un més de cas extrem:
# 
# <img src="img/page3.png" />

# In[4]:

if __name__ == '__main__':
    G3 = np.asarray((
        (0,   0, 1/2.0, 0), 
        (0,   0, 1/2.0, 0), 
        (1.0, 0, 0,     0), 
        (0,   0, 0,     0)
    ))
    #x = power_method(G3)
    y = solve_eig(G3)
    
    print(np.round(G3, 2))
    print('Eigenvector', np.round(y, 2))
    print('Eigenvalues', np.round(np.linalg.eigvals(G3), 2))


# ## Solucions
# 
# Per evitar tenir cicles, graphs separats i dangling nodes, el que es fa és modificar la matriu G amb "soroll", per tal de que tot quedi connectat amb tot. Aquesta tècnica a vegades rep el nom de "Random Surfer".
# 
# $$M = (1 - m)G + mS$$
# 
# On $S$ és una matriu amb totes les entrades $1/n$ i $m$ un nombre petit, normalment $0.15$.

# **Fes una funció que donada la matriu $G$ i $m$ calculi la nova matriu $M$**

# In[5]:

def fix_matrix(G, m=0.15):
    #"Arreglem G"
    dimensioG = G.shape[0]
    S = np.ones((dimensioG,dimensioG))/dimensioG #Calcul de S
    return((np.dot(G,(1-m))+(np.dot(S,m)))) #Apliquem la formula


# In[6]:

if __name__ == '__main__':
    M2 = fix_matrix(G2)
    x = power_method(M2)
    y = solve_eig(G2)
    z = solve_eig(M2)
    
    print(np.round(M2, 2))
    print('Eigenvector', np.round(x, 2))
    print('Eigenvector', np.round(y, 2))
    print('Eigenvector', np.round(z, 2))
    print('Eigenvalues', np.round(np.linalg.eigvals(M2), 2))


# In[7]:

if __name__ == '__main__':
    M3 = fix_matrix(G3)
    x = power_method(M3)
    y = solve_eig(G3)
    z = solve_eig(M3)
    
    print(np.round(M3, 2))
    print('Eigenvector', np.round(x, 2))
    print('Eigenvector', np.round(y, 2))
    print('Eigenvector', np.round(z, 2))
    print('Eigenvalues', np.round(np.linalg.eigvals(M3), 2))


# # Recomanant
# 
# El primer que haurem de fer és construir una matriu que ens serveixi, d'alguna forma, com a indicatiu de preferències de cada persona. Per tal efecte, construirem una matriu $m\times n$, de $m$ usuaris per $n$ items, on cada entrada $i,j$ serà el nombre de vegades que la persona $i$ a comprat l'item $j$.
# 
# <img src="img/Mat.png">
# 
# Per saber de quin usuari és cada `order_id`, haureu de creaur el dataset `order_products` amb el `orders`. Una sola persona/usuari tindrà més d'una ordre, mireu quants cops ha comprat els mateixos productes.
# 
# A més, les dades es composen de molts `product_id` diferents, hi ha massa diversitat entre usuaris. Per tant, per poder recomanar el que farem serà agregar les dades, enlloc de treballar per `product_id` ho farem per `aisle_id`, és a dir "la secció" del súper on es troba.
# 
# Al llarg de la pràctica es parlarà de producte i/o item, doncs és la terminologia estàndard de recomanadors, però sempre serà en referència a `aisle_id` per aquesta pràctica!

# In[8]:

import zipfile
from os.path import join, dirname

def locate(*path):
    base = globals().get('__file__', '.')
    return join(dirname(base), *path)

def unzip(file):
    zip_ref = zipfile.ZipFile(locate(file), 'r')
    zip_ref.extractall(locate('data'))
    zip_ref.close()

unzip('order_products__train.csv.zip')
unzip('orders.csv.zip')
unzip('products.csv.zip')


# In[9]:

import pandas as pd

if __name__ == '__main__':
    df_order_prods = pd.read_csv(locate('data', 'order_products__train.csv'))
    df_orders = pd.read_csv(locate('data', 'orders.csv'))[['order_id', 'user_id']]
    df_prods = pd.read_csv(locate('data', 'products.csv'))[['product_id', 'aisle_id']]


# In[ ]:

print(df_order_prods)
print(df_orders)


# In[11]:

if __name__ == '__main__':
    ### Creua df_order_prods i df_orders
    df_merged = df_order_prods.set_index('order_id').join(df_orders.set_index('order_id'))
    #Ara user id ajunta-ho amb product_id i aisle_id
    df_merged = pd.merge(df_merged, df_prods)[['user_id','product_id','aisle_id']]
    print(df_merged)


# Fes la funció que retorna els productes comprats en cada `aisle_id` per cada `user_id`.

# In[12]:

def build_counts_table(df):
    """
    Retorna un dataframe on les columnes són els `aisle_id`, les files `user_id` i els valors
    el nombre de vegades que un usuari ha comprat un producte d'un `aisle_id`
    
    :param df: DataFrame original després de creuar-lo
    :return: DataFrame descrit adalt
    """
    df = df.groupby(['user_id','aisle_id']).size() #Agrupa o "ordena" el DataFrame segons cada usuari i el seu aisle_id
    return df.unstack().fillna(0) #Unstack: per pivotar el DataFrame en columnes. Fillna: posar un zero a on hi hagi un NaN

def get_count(df, user_id, aisle_id):
    """
    Retorna el nombre de vegades que un usuari ha comprat en un `aisle_id`
    
    :param df: DataFrame retornat per `build_counts_table`
    :param user_id: ID de l'usuari
    :param aisle_id: ID de la secció
    :return: Enter amb el nombre de vegades que ha comprat
    """
    
    return df.loc[user_id][aisle_id] #Retorna la casella ubicada per user_id i el seu aisle_id


# In[13]:

if __name__ == '__main__':
    df_counts = build_counts_table(df_merged)
    count = get_count(df_counts, 14, 5)
    print(count)


# Tenim moltes dades en el nostre dataset, pel que és convenient que les reduïm una mica. Per començar a treballar recomanem que reduiu el tamany a aproximadament 0.001 de l'original (`frac=0.001`). Podeu provar, més endavant, amb 0.01.
# 
# A més, necessitem poder provar quan bé funciona el nostre sistema. Pel que dividirem les dades de cada usuari en 2 parts:
# 
# 1. **Train**: Els items que farem servir per entrenar el nostre recomanador
# 2. **Test**: Dades "ocultes" que ens serviran per provar quant bé funciona el sistema
# 
# **Nota** Pot tardar bastant aquesta cel·la!

# In[14]:

from sklearn.model_selection import train_test_split

def split_train_test(df):
    """
    No modifica l'estructura del DataFrame original,
    únicament el divideix en 2 sub-DataFrame's.
    
    Tots dos tenen el mateix nombre d'usuaris, però cada
    un té un conjunt diferent de producte d'aquest
    
    :param df: DataFrame retornat per `build_counts_table`
    :return: Dos DataFrames amb diferents productes
    """
    split = lambda i: (
        train_test_split(row[row > 0], test_size=0.3, random_state=uid)[i] \
        for uid, row in df.iterrows()
    )
    train = pd.DataFrame(split(0)).fillna(0)
    test = pd.DataFrame(split(1)).fillna(0)
    return train, test

if __name__ == '__main__':    
    df_counts_train, df_counts_test = split_train_test(df_counts)


# In[15]:

if __name__ == '__main__':
    FRAC = 0.001
    df_reduced_counts_train = df_counts_train.sample(frac=FRAC, random_state=1)
    df_reduced_counts_test = df_counts_test.sample(frac=FRAC, random_state=1)
    
    print(df_reduced_counts_test.shape)
    print(df_reduced_counts_train.shape)


# # Graph ampliat
# 
# Si ara construíssim un graph com el que fèiem per les webs i generessim el vector principal, obtindríem efectívament un vector amb la importància de cada persona... però, relativa a que?
# 
# <img src="img/Matvs.png">
# 
# Per tal de solucionar aquest problema, on no sabem que és que del pagerank resultant, el que farem serà ampliar el graph, en certa forma duplicant la informació que tenim. 
# 
# <img src="img/Matext.png">
# 
# Hauràs de construir una matriu $m+n \times n+m$, on:
# 
# * Les $m$ primeres files i les $n$ últimes columnes (indexos $0,m$) sigui la matriu que has construit anteriorment **normalitzada**
# * Les últimes $n$ files i les primeres $m$ columnes (indexos $m,0$) sigui la matriu anterior però transposada i **normalitzada**
# * La resta d'entrades, 0
# 
# 
# **normalitzada**: Aquesta matriu $m\times n$ ha d'estar normalitzada per columnes (les columnes han de sumar 1). Per simplificar les imatges i que siguin més entenedores, es fan servir els valors reals. Però és molt important que normalitzeu en el vostre codi!

# In[16]:

#Funcio que pasada una matriu, la normalitza "manualment", ja que .norm ens donava problemes.
def normalitzar(matriu):
    vector = np.array(matriu.sum(axis=0)).ravel()
    vector[vector>0]=1 / vector[vector>0]
    vector = np.diag(vector)
    matriu = matriu.dot(vector)
    return matriu


# In[17]:

from numpy.linalg import norm 
def get_extended_graph(df_train):
    """
    Calcula el graf ampliat de prodcutes i usuaris a partir de l'original
    
    :param df: Sub-DataFrame del retornat per `build_counts_table`, per training
    :return: El graf ampliat, tal i com està descrit adalt, tenint en compte
        que la suma de cada columna ha de ser 1 (normalitzar les columnes, és a dir
        dividir cada número de la columna pel total de la suma de la mateixa columna)
    """  
    m,n = df_train.shape
    matriu_extended = np.zeros((m+n,n+m)) #Creem la nova matriu amb les dimensions adecuades
    matriu_extended[:m,m:] = df_train.values #Posem els valors de la matriu en els indexos 0,m de columnes
    matriu_extended[m:,:m] = df_train.values.T #Ara, transposats i en columnes m, 0
    
    matriu_final = normalitzar(matriu_extended) #Normalitzem
    return matriu_final


# In[18]:

if __name__ == '__main__':
    G = get_extended_graph(df_reduced_counts_train)


# ## Recomanació personalitzada
# 
# 
# Seguim tenint un altre problema, i és, estem personalitzant res? La matriu ampliada és exactament la mateixa per cada usuari, independentment del que hagi comprat, i per tant el resultat serà sempre el mateix.
# 
# Suposa que volem recomanar a l'usuari 1, el que farem serà crear una altre matriu del mateix tamany que l'anterior, on tots els elements seran 0 excepte aquelles files i columnes (corresponents a la matriu ampliada anterior) dels items que ha comprat l'usuari.
# 
# <img src="img/Matper.png">

# Finalment, la matriu sobre la qual farem el càlcul de pagerank serà la matriu ampliada perturbada per aquesta nova matriu.
# 
# Anomena $G$ a la original i $E$ a aquesta que acabes de fer, i $\bar{E}$ és $E$ normalitzada per columnes, la matriu final $G_m$ serà:
# 
# $$G_m = (1-m)G + m\bar{E}$$
# 
# Que ja us hauria de sonar! Ho hem fet abans amb pagerank. Fixeu-vos que per cada usuari la matriu $G_m$ serà diferent, doncs tot i que $G$ no canvia, sí que ho fa $E$

# In[23]:

def personalize(G, df_train, user, m=0.15):
    """
    Personalitza el graf ampliat per a un usuari donat.
    
    La matriu E, un cop construida i abans de fer-la servir per personalitzar G,
    s'ha de normalitzar per columnes.
    
    :param G: El graf ampliat
    :param df: Sub-DataFrame del retornat per `build_counts_table`, per training
    :param user: ID d'usuari
    :param m: Valor de perturbació, tal i com està descrit adalt
    :return: Matriu ampliada personalitzada
    """
    
    indexs = df_train.loc[user].loc[df_train.loc[user] != 0].index #Agafem els items comprats per l'user
    e = np.zeros_like(G) #Creem E
    e[df_train.shape[0]+indexs,:] = 1 #Posem un 1 en aquets indexs
    e[:,df_train.shape[0]+indexs] = 1
    E = normalitzar(e) #Normalitzem
    #I apliquem la formula
    return (1-m)*G+m*E


# In[24]:

if __name__ == '__main__':
    Gm = personalize(G, df_reduced_counts_train, 93427)


# Finalment, ara que ja tenim $G_m$, podem executar pagerank i obtenir el vector principal. Com pots observar a la última imatge, aquest vector tindrà $m+n$ elements, els primers $m$ corresponents als usuaris i els següents $n$ als items.
# 
# Com que volem similituds entre usuaris, ens quedarem solament amb la primera part, fins a $m$. A més, el propi usuari a qui hem personalitzat la matriu no l'hem de tenir en compte, així que cal posar-l'ho a 0.

# In[25]:

def sims_vect(Gm, df_train, user):
    """
    Calcula el vector de similituds per a un usuari donat, és a dir
    executa el metòde de la potència sobre el graf ampliat personalitzat
    de l'usuari, i en retorna els primers M elements del vector resultant.
    
    A més, posa a 0 la posició del vector corresponent a l'usuari al que
    estem recomanant.
    
    :param Gm: Graf ampliat personalitzat
    :param df: Sub-DataFrame del retornat per `build_counts_table`, per training
    :param user: ID de l'usuari
    :return: Vector de similituds en una array de numpy
    """
    #Creem un diccionari on lliguem l'index del usuari al Dataframe 
    diccionari = dict(zip(df_train.index, range(df_train.shape[0])))
    similitud = power_method(Gm) #Metode de la potencia per calcular la similitud
    similitud = similitud[:df_train.shape[0]] #Volem similituds entre usuaris, aixi que ens quedem nomes fins a m
    similitud[diccionari[user]] = 0 #No em de tenir en compte el propi usuari de qui hem personalitzat la matriu
    
    return similitud
    


# In[26]:

if __name__ == '__main__':
    sims = sims_vect(Gm, df_reduced_counts_train, 93427)


# Ara aplica la formula per recomanacions colaboratives donat un usuari $u$ i item $i$
# 
# $$\hat{r}_{u,i} = \frac{\sum_{p,p\neq u} sim(u,p) \cdot r_{p,i}}{\sum_p sim(u,p)}$$
# 
# Tingues en compte que aquesta fòrmula solament té en compte aquells usuaris que també han comprat el mateix! Si no han comprat, no s'ha de comptar en el sumatori!

# In[27]:

def score(df_train, user, item, sims):
    """
    Fent servir la fòrmula del filtratge colaboratiu, retorna un valor
    per a un usuari i producte
    
    :param df: Sub-DataFrame del retornat per `build_counts_table`, per training
    :parma user: ID de l'usuari
    :param item: ID de l'item
    :param sims: Vector de similituds per a l'usuari
    :return: Un flotant indicant el valor computat segons la fòrmula d'adalt
    """
    items = df_train.loc[:,item] > 0 #Agafem els items comprats
    superior = sims[items].dot(df_train.loc[items,item]) #Multipliquem (part de dalt de la formula)
    inferior = sims[items].sum() #Part inferior de la formula
    
    if np.allclose(inferior, 0): #Tots els valors propers a zero (Molt petits) els posem a zero millor
        return 0
    
    R = superior/inferior #Dividim
    return R

    


# In[28]:

if __name__ == '__main__':
    print(score(df_reduced_counts_train, 93427, 98, sims))
    print(df_reduced_counts_test.loc[93427, 98])
    print(df_reduced_counts_train.loc[93427, 98])


# Ara, donat un usuari, recomana-li els $k$ millors productes que podria comprar. Per fer-ho, computa l'`score` per a cada possible item que encara no hagi comprat, ordena i retorna els $k$ millors.
# 
# Si $k=0$, significa retornar tots els possibles.

# In[29]:

def recommend(df_train, sims, user, k, score=score):
    """
    Calcula l'`score` de tots els items d'un usuari, que no hagin estat ja comprats,
    i retorna els $k$ amb valor més alt. Si $k=0$, els retorna tots ordenats de major a menor.
    
    :param df: Sub-DataFrame del retornat per `build_counts_table`, per training
    :param sims: Vector de similituds
    :param user: ID de l'usuari
    :param k: Número de valors a retornar, o 0 per tots
    :param score: Funció a fer servir per calcular l'score
    :return: Llista amb els $k$ (o tots si $k=0$) items més alts. Cada element d'aquest
        vector serà una tupla (valor, aisle_id)
    """
    #Aqui hem de trobar els items amb valor mes alt, els que recomanarem. Per fer-ho, fem:
    #Trobem els items no comprats per l'usuari (agafem els seus indexs)
    #Per cada un d'ells, els hi calculem la seva score.
    #Fiquem a una llista parelles (puntuacio, index del item) per cada item no comprat
    #Ordenem aquesta llista segons la puntuacio, de petit a gran. La girem, perque volem els items amb major puntuacio
    #Ara em de retornar k items amb major puntuacio, aixi que anem afegint a una llista final cada parella (puntuacio,item)
    
    llista = [] 
    noComprats = df_train.loc[user].loc[df_train.loc[user] == 0].index
    for i in noComprats:
        puntuacio = score(df_train, user, i, sims)
        llista.append((puntuacio,i))
    llista.sort(key=lambda x: x[0]) #Ordenar per puntuacio
    llista = llista[::-1] #La girem (de gran a petit)
    
    llistaFinal = []
    if k != 0:
        for i in range(k):
            llistaFinal.append(llista[i])
        return llistaFinal
    else:
        return llista


# In[30]:

if __name__ == '__main__':
    print(recommend(df_reduced_counts_train, sims, 93427, 5))


# ## Resultats
# 
# * No hem pogut comprovar els resultats amb molta gent, només amb un altre grup.
# * No coincideixen amb els del professor, pero si que amb aquest altre grup hem tingut pràcticament els mateixos resultats.
# * De fet, tenim els mateixos productes, pero amb una puntuacio que varia en varis decimals.
# * En conseqüència, les funcions Evaluant donen valors molt diferents si comparem amb els resultats del professor i els mateixos comparant amb aquests altre grup:

# ## Evaluant
# 
# Per saber si em fet un bon recomanador, hem d'avaluar si està funcionant correctament. Ho farem predint la puntuació de tots els items de test per un usuari, i comparant amb els valors reals.
# 
# Les funcions ja estan fetes, simplement podeu executar per veure que us surt un nombre raonable.

# In[31]:

def evaluate(df_train, df_test, sims, user, score=score):
    real = df_test.loc[user]
    pred_list = recommend(df_train, sims, user, 0, score=score)
    pred = pd.Series({y: x for x, y in pred_list})
    
    real = real[real > 0]
    pred = pred.loc[real.index]
    
    return np.sum(np.power(real - pred, 2))
        

def mean_eval(df_train, df_test, users, score=score):
    G = get_extended_graph(df_train)
    return np.mean([
        evaluate(
            df_train, 
            df_test, 
            sims_vect(personalize(G, df_train, uid), df_train, uid), 
            uid,
            score=score
        ) \
        for uid in users if df_train.loc[uid].sum() > 0
    ])


# In[32]:

if __name__ == '__main__':
    print(evaluate(df_reduced_counts_train, df_reduced_counts_test, sims, 93427))


# In[33]:

if __name__ == '__main__':
    users = df_reduced_counts_test.sample(n=10).index
    print(mean_eval(df_reduced_counts_train, df_reduced_counts_test, users))


# # Propostes de millora

# ## 1. Millorar la recomanació colaborativa
# 
# La fòrmula que fem servir per calcular l'`score`, basada en la recomanació colaborativa, és força inexacta, doncs no té en compte el *bias* introduit per la mitja del comprador.
# 
# Per exemple, jo potser tinc família numerosa i compro sempre 5 del mateix producte com a mínim, mentre que algú que visqui sol únicament en compraria 1 unitat.
# 
# Podem eliminar aquest bias fent:
# 
# $$\hat{r}_{u,i} = \frac{\sum_{v,v\neq u} sim(u,v) \cdot (r_{v,i} - \mu_v)}{\sum_{v,v\neq u} sim(u,v)} + \mu_u$$
# 
# És a dir, a cada producte d'altres persones li restem la mitja d'aquella persona ($\mu_v$) i al final reintroduïm la mitja de l'usuari a qui estem recomanant ($\mu_u$)

# In[ ]:

def score_mean(df_train, user, item, sims):
    raise NotImplementedError()


# In[ ]:

if __name__ == '__main__':
    print(score_mean(df_reduced_counts_train, 93427, 98, sims))


# In[ ]:

if __name__ == '__main__':
    mean_eval(df_reduced_counts_train, df_reduced_counts_test, users, score=score_mean)


# **Nota**: Recalcular les mitges cada cop és molt lent, probablement voldràs tenir-les precalculades (amb una variable global o semblant)

# ## 2. Utilitzar més dades

# Una simple inspecció de la taula de comptes, ens mostrarà que un gran percentatge està buida:

# In[ ]:

if __name__ == '__main__':
    total = np.prod(df_counts.shape)
    zero = (df_counts == 0).sum().sum()
    nonzero = total - zero

    print('Are 0: {:2.3g}%'.format(zero / total * 100))


# Per tant, podríem pensar que enlloc de guardar absolutament tot, solament necessitem saber les posicions on no hi ha 0's i el seu valor. Això és precisament el que fan les estructures de la llibreria
# 
# ```python
# import scipy.sparse as sparse
# ```
# 
# Tot el contingut de la llibreria `sparse` són presentacions no denses de matrius, únicament guarden els elements que són diferents de 0.
# 
# Es proposa que canvieu les funcions `get_extended_matrix` i `personalize` per tal de que facin servir `sparse.lil_matrix` enlloc de `np.array`, i que augmenteu `FRAC` a `1.0` (totes les dades).

# **Nota**: No es tracta d'un procés trivial, treballar amb matrius sparse té cert misteri. 
# 
# **1.** No feu servir mai funcions de numpy sobre una matriu sparse, per exemple, supossa que `mat` és una `sparse.lil_matrix` i `dia` una matriu diagonal també `sparse.lil_matrix`:
# 
# ```python
# res = np.dot(mat, dia)
# ```
# 
# Farà el producte matricial, però la matriu resultant `res` no serà sparse sinó densa. Feu servir sempre la versió "metòdica" de les funcions:
# 
# ```python
# res = mat.dot(dia)
# ```
# 
# I ara, `res` és sparse, tal i com s'espera.
# 
# ------------------
# 
# **2.** Si vols agafar una columna sencera i convertir-la a un vector dens, tal i com faria numpy, no és suficient fent:
# 
# ```python
# col = mat[:,0]
# ```
# 
# Doncs el resultat, no és un vector de $n$ elements, sinó una matriu $n,1$. Podríem pensar que aplanar el vector resultaria:
# 
# ```python
# col = mat[:,0].flatten() # O, equivalentment, mat[:,0].ravel()
# ``` 
# 
# Però tampoc funcionarà. Cal convertir explícitament a array de numpy abans:
# 
# ```python
# col = np.array(mat[:,0]).ravel() # o flatten()
# ```
# 
# I, ara sí, tenim un vector de $n$ elements.
