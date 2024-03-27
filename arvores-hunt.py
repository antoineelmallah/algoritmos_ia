from math import log2, pow
import numpy as np

'''
Dt = conjunto de instâncias
y = {y1,y2,...,yc} rotulos das classes
1 - Se todas as instâncias Dt pertencem a mesma classe yi, t é nodo folha
2 - Se Dt não é homogêneo, selecionar o melhor atributo, criar novo nodo 
e dividir as instâncias de acordo com os valores dos atributos, recursivamente.
'''

class Calc:

    def __init__(self, target):
        self.target = target
        self.true = 0
        self.false = 0

    def increment(self, target):
        if target == True:
            self.true = self.true + 1
        else:
            self.false = self.false + 1

    def __str__(self) -> str:
        return 'true: ' + str(self.true) + ', false: ' + str(self.false)
    
    def impurity(self):
        total = self.total()
        if total == 0:
            return 0
        result = 1 - pow(self.true/total, 2) - pow(self.false/total, 2)
        print('gini:', result)
        return result
    
    def total(self):
        return self.true + self.false


class FeatureCalc:

    def __init__(self, idx_feature, features, targets):
        self.idx_feature = idx_feature
        self.left = Calc(True)
        self.right = Calc(False)
        for idx, ft in np.ndenumerate(features == targets):
            if features[idx] == True:
                self.left.increment(ft)
            else:
                self.right.increment(ft)

    def __str__(self) -> str:
        return str(self.idx_feature) + ' => left: ' + str(self.left) + ', right: ' + str(self.right)
    
    def impurity(self):
        result = (self.left.total() / (self.left.total() + self.right.total()) * self.left.impurity()) + (self.right.total() / (self.left.total() + self.right.total()) * self.right.impurity())
        return result
    
class AbstractNode:
    pass

class Node(AbstractNode):
    def __init__(self, trueNode: AbstractNode, falseNode: AbstractNode):
        super().__init__()
        self.true = trueNode
        self.false = falseNode

    def __str__(self) -> str:
        return 'NODE => \n- left: ' + self.true + ', \n- right: ' + self.false

class Leaf(AbstractNode):
    def __init__(self, nodeClass: bool):
        super().__init__()
        self.nodeClass = nodeClass
    
    def __str__(self) -> str:
        return 'LEAF => class: ' + self.nodeClass


def tree(Dt, y):
    np_dt = np.array(Dt)
    np_y = np.array(y)
    (idx_lower_impurity, lowest_impurity) = get_lower_impurity_feature_idx(np_dt, np_y)
    if lowest_impurity > 0:
        np_true = [ x for x in np_dt if x[idx_lower_impurity] == True ]
        y_true = [ y for idx, y in np.ndenumerate(targets) if np_dt[idx, idx_lower_impurity] == True ]
        left = tree(np_true, y_true)
        np_false = [ x for x in np_dt if x[idx_lower_impurity] == False ]
        y_false = [ y for idx, y in np.ndenumerate(targets) if np_dt[idx, idx_lower_impurity] == False ]
        right = tree(np_false, y_false)
        return Node(left, right)
    else:
        return Leaf(np_dt[0, idx_lower_impurity])


def get_lower_impurity_feature_idx(np_dt, np_y):
    lowest_impurity = None  
    idx_lower_impurity = None
    for idx_feature in range(np_dt.shape[1]):
        features = np_dt[:, idx_feature]
        calc = FeatureCalc(idx_feature, features, np_y)
        impurity = calc.impurity()
        if lowest_impurity == None or lowest_impurity > impurity:
            lowest_impurity = impurity
            idx_lower_impurity = idx_feature
    return (idx_lower_impurity, lowest_impurity)


Dt = [
    [True,  True  ], 
    [True,  False ], 
    [False, True  ], 
    [False, True  ], 
    [True,  True  ], 
    [True,  False ], 
    [False, False ]]

targets = [False, False, True, True, True, False, False]

t = tree(Dt, targets)

print(t)