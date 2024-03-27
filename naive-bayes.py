import pandas as pd

X = pd.DataFrame([
    ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'], 
    ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild'], 
    ['High','High','High','High','Normal','Normal','Normal','High','Normal','Normal','Normal','High','Normal','High'], 
    [False,True,False,False,False,True,True,False,False,False,True,True,False,True]]).transpose()

features = ['outlook', 'temperature', 'humidity', 'windy']

X.columns = features

X['target'] = ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

def count_or_zero(series, value):
    return series[value] if value in series else 0

def prob_A_dado_B(feature, f_value, t_value):
    feature_unique_values = X[feature].unique()
    count_features = count_or_zero(X[X[feature].isin(feature_unique_values)].groupby(['target'])['target'].count(), t_value)
    count_feature = count_or_zero(X[X[feature] == f_value].groupby(['target'])['target'].count(), t_value)
    '''
    Foi somado 1 em cada contagem de feature (numerador). Por isso, deve ser adicionado 1 para 
    cada feature (len(feature_unique_values)), no denominador.
    Isso deve ser feito para nunca uma das probabilidades do produtório ser zero. Isso faria
    com que todo o produtório fosse zerado.
    '''
    return (count_feature + 1) / (count_features + len(feature_unique_values))

#print(prob_A_dado_B(feature='outlook', f_value='Rainy', t_value='No'))

def prob_y(data, y):
    # probabilidade a priori
    count_y = data.groupby(['target'])[data.columns[0]].count()[y]
    size = len(data)
    return count_y / size

def cond_prob_prod(x, y):
    # likelyhood
    result = 1
    for idx, feature in enumerate(features):
        p = prob_A_dado_B(feature, x[idx], y)
        result = result * p
    #print('cond_prob_prod', result)
    return result

'''
# Sempre igual!!!
def feat_prob_prod(data, x):
    # probabilidade a posteriori
    result = 1
    size = len(data)
    for idx, feature in enumerate(features):
        f_value = x[idx]
        count_f_value = count_or_zero(data.groupby(feature)[feature].count(), f_value)
        p = count_f_value / size
        #print(p)
        result = result * p
    #print('feat_prob_prod', result)
    return result
'''

def naive_bayes(data, x):
    targets = data['target'].unique()
    prob = 0
    result = None
    for y in targets:
        y_prob = prob_y(data, y) * cond_prob_prod(x, y) # / feat_prob_prod(data, x) # sempre igual!!!
        print(y, y_prob)
        if y_prob > prob:
            prob = y_prob
            result = y
    return result

print(naive_bayes(data = X, x = ['Sunny', 'Cool', 'High', True]))