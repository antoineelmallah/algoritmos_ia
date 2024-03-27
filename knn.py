import pandas as pd
from math import sqrt, pow, fabs

'''
K-NN
- K define a complexidade do modelo. Para menores valores de K, modelo mais complexo
- K muito baixo (menos viés, mais variância - overfitting)
- K muito alto (mais viés, menos variância - underfitting)
- Classificador LAZY: não construimos um modelo
- Pré-processamento excencial
- Processo de predição para novos objetos custoso
- Não é interpretável em nível global
- Possui algoritmo simples
- Produz fronteira não linear (diagrama de Voronoi)
- Sofre influência de atributos irrelevantes ou redundantes
- É impactado pela alta dimensionalidade de dados (muitos atributos)
- Pode ser usado para classificação ou regressão
- Sensível às unidades de medida dos atributos (solução: normalização ou standartização)
'''

def knn(X, y, dist, xt, k):
    distances = pd.DataFrame(data=y, columns=['target'])
    distances['distance'] = [dist(x,xt) for x in X]
    distances = distances.sort_values(by='distance', ascending=True)
    print(distances)
    kdist = distances[:k].groupby(['target'])['distance'].count()
    if kdist.size == 0:
        return None  
    if kdist.size == 1:
        return kdist.keys()[0]
    max_value = max(kdist.values)
    return [x for x in kdist.keys() if kdist[x] == max_value][0]

def euclidian(v1, v2):
    verify_lenght(v1, v2)
    result = 0
    for i in range(len(v1)):
        result = result + pow(v1[i] - v2[i], 2)
    return sqrt(result)

def manhatan(v1, v2):
    verify_lenght(v1, v2)
    result = 0
    for i in range(len(v1)):
        result = result + fabs(v1[i] - v2[i])
    return result

def verify_lenght(v1, v2):
    if len(v1) != len(v2):
        raise BaseException('Vectors should have same size')




#################################################################

X = [
    [14,23],[15,28],[15,20],[16,21],[20,22],[15,14],
    [9,21],[9,14],[11,19],[9,17],[13,16]
    ]

y = ['Malignant','Malignant','Malignant','Malignant','Malignant','Malignant',
     'Benign','Benign','Benign','Benign','Benign']

result = knn(X, y, euclidian, [13,18], 2)
print('RES:', result)

result = knn(X, y, manhatan, [13,18], 2)
print('RES:', result)
