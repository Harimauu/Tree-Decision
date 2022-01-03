# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:11:31 2021

@author: wikas
"""

import pandas as pd
# Для отсутствия лишних сообщений в pd
pd.options.mode.chained_assignment = None
# Критерий Пирсона 
from scipy.stats import pearsonr
import math
# ROC-кривая
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
# Для игнорирования предупреждений
#import warnings
#warnings.filterwarnings("ignore")

# Функция для проверки, является ли значение числом
def isDigit(val):
    if type(val) == str:
        if val.isdigit():
           return True
        else:
            try:
                float(val)
                return True
            except:
                return False
    else: return True

# Изменение типа на бинарный
def toBin(att, rvs=False):
    attBin = att.unique()
    if rvs == True:
        attBin[0], attBin[1] = attBin[1], attBin[0]
    for i in range(att.size):
        if att[i] == attBin[0]:
            att[i] = 0
        else: att[i] = 1
    return att

# Сортировка атрибутов
def sortAtts(atts, tagAtt, criterion="entropy", attsVals=None):
    # Словарь для хранения названия атрибута и статистики критерия
    attsDict = {}
    for i in range(atts.columns.size):
        if criterion == "pearson":  
            stat = pearsonr(atts[atts.columns[i]], tagAtt)[0]
        else:
            stat = compEntropyAndGini(tagAtt, atts[atts.columns[i]], criterion=criterion, attsVals=attsVals, stat=True)
        attsDict[atts.columns[i]] = stat
    # Отсортированный словарь
    sortedAtts = {}
    # Сортировка ключей по значениям
    sortedKeys = sorted(attsDict.items(), key=lambda x: math.fabs(x[1]), reverse=True)
    for i in sortedKeys:
        sortedAtts[i[0]] = attsDict[i[0]]
    return sortedAtts

# Выбор атрибута
def selAtt(sortedAtts, auto=True):
    if auto == True:
        return list(sortedAtts.keys())[0]
    else:
        print(f'Доступные атрибуты: {sortedAtts}')
        while True:
            att = input("Выберите атрибут: ")
            if att in sortedAtts:
                return att
            else:
                print("Такого атрибута нет")
            
# Является ли блок листом
def isLeaf(p):
    if p == 0.0 or p == 1.0:
        return True
    else: return False
            
# У всех ли объектов один и тот же результат            
def sameRes(tagAtt, indexes):
    for i in indexes:
        if tagAtt[i] != tagAtt[indexes[0]]:
            return False
    return True

# Разбиение числовых атрибутов на границы
def setPartitionBoundary(tagAtt, selAtt, indexes, pred=True, condition=None, criterion="entropy"):
    # Если дерево решений уже построено...
    if pred == True:
        bound = float(condition)
    # Если дерево решений только строится...
    else:
        # ...создаётся словарь для хранения объектов,
        data = {}
        for i in indexes:
            data[selAtt[i]] =  tagAtt[i]
        # элементы сортируются,
        x = list(selAtt)
        x.sort()
        y = []
        for i in range(len(x)):
            y.append(data[x[i]])
        # создаются списки для хранения двух частей объектов
        lstX = [[], []]
        lstY = [[], []]
        # Максимальное значение, полученное по критерию
        maxVal = 0
        # Граница разбиения
        bound = 0
        if criterion == "pearson":
            # ...если объектов >= 4...
            if len(indexes) >= 4:
                # Разбиение по коэффициенту корреляции Пирсона
                for i in range(2, len(x)-1):
                    lstX[0], lstY[0], lstX[1], lstY[1] = x[:i], y[:i], x[i:], y[i:]
                    # Сумма коэффициентов корреляции Пирсона, умноженных на взвешенные коэффициенты
                    val = (math.fabs(pearsonr(lstX[0], lstY[0])[0]) * (len(lstX[0]) / len(x))) + (math.fabs(pearsonr(lstX[1], lstY[1])[0]) * (len(lstX[1]) / len(x)))
                    if maxVal < val:
                        maxVal = val
                        bound = x[i]
            # ...если объектов < 4...
            else: bound = selAtt[indexes[1]]
        else:
            bound = compEntropyAndGini(tagAtt, selAtt, indexes, criterion)
    # Разбиение объектов на 2 границы
    trueLst = []
    falseLst = []
    for i in indexes:
        if selAtt[i] < bound:
            trueLst.append(i)
        else:
            falseLst.append(i)
    # Если дерево уже построено...
    if pred == True:    
        return trueLst, falseLst
    # Если дерево только строится...
    else:
        return trueLst, falseLst, f"< {bound}"

# Разбиение номинального атрибута на 2 границы
def setAttVal(selAtt, indexes, attNames=None, pred=True, val=None, auto=True):
    trueLst = []
    falseLst = []
    # Если дерево решений только строится...
    if pred == False:
        # ...если дерево строится автоматически...
        if auto:
            count = [0, 0]
            for i in indexes:
                if selAtt[i] == 0:
                    count[0] += 1
                else: count[1] += 1
            attVal = attNames[0] if count[0] >= count[1] else attNames[1]
        # ...иначе, если дерево строится поэтапно
        else:
            while True:
                attVal = input(f"Выберите {attNames[0]} или {attNames[1]}: ")
                if attVal != attNames[0] and attVal != attNames[1]:
                    print("Неверное значение")
                else: break
        if attVal == attNames[0]:
            val = 0
        else: 
            val = 1
    for i in indexes:
        if selAtt[i] == val:
            trueLst.append(i)
        else:
            falseLst.append(i)
    # Если дерево решений только строится...
    if pred == False:
        return trueLst, falseLst, attVal
    # ...иначе возвращаются только границы (без условия)
    else: return trueLst, falseLst

# Расчёт энтропии
def compEntropyAndGini(tagAtt, selAtt, indexes=None, criterion="entropy", attsVals=None, stat=False):
    if indexes == None:
        indexes = tagAtt.index.tolist()
    # Если атрибут - НЕ номинальный
    try:
        attsVals[selAtt.name]
    except:
        # ...создаётся словарь для хранения объектов,
        data = {}
        for i in indexes:
            data[selAtt[i]] =  tagAtt[i]
        # элементы сортируются,
        x = list(selAtt)
        x.sort()
        y = []
        for i in range(len(x)):
            y.append(data[x[i]])
        # создаются списки для хранения двух частей объектов
        lstY = [[], []]
        # Максимальное значение, полученное по критерию
        maxVal = 0
        # Граница разбиения
        bound = 0
        for i in range(1, len(x)):
            # Количество true и false в первом списке
            countY0 = [0, 0]
            # Количество true и false во втором списке
            countY1 = [0, 0]
            lstY[0], lstY[1] = y[:i], y[i:]
            for j in range(len(lstY[0])):
                if lstY[0][j] == 0:
                    countY0[0] += 1
                else: countY0[1] += 1
            for j in range(len(lstY[1])):
                if lstY[1][j] == 0:
                    countY1[0] += 1
                else: countY1[1] += 1
            if criterion == "gini":
                S1 = 1 - ((countY0[0] / len(lstY[0])) ** 2) - ((countY0[1] / len(lstY[0])) ** 2)
                S2 = 1 - ((countY1[0] / len(lstY[1])) ** 2) - ((countY1[1] / len(lstY[1])) ** 2)
                val = 1 - ((len(lstY[0]) / len(y)) * S1) - ((len(lstY[1]) / len(y)) * S2)
            else:
                # Если хоть одна вероятность равна нулю, то и энтропия тоже равна нулю
                if countY0[0] == 0 or countY0[1] == 0:
                    S1 = 0
                else:
                    S1 = -(countY0[0] / len(lstY[0])) * math.log(countY0[0] / len(lstY[0]), 2) - (countY0[1] / len(lstY[0])) * math.log(countY0[1] / len(lstY[0]), 2)
                if countY1[0] == 0 or countY1[1] == 0:
                    S2 = 0
                else:
                    S2 = -(countY1[0] / len(lstY[1])) * math.log(countY1[0] / len(lstY[1]), 2) - (countY1[1] / len(lstY[1])) * math.log(countY1[1] / len(lstY[1]), 2)
                val = 1 - ((len(lstY[0]) / len(y)) * S1) - ((len(lstY[1]) / len(y)) * S2)
            if maxVal < val:
                maxVal = val
                bound = x[i]
    # Если атрибут - номинальный
    else:
        lstY = [[], []]
        for i in indexes:
            if selAtt[i] == 0:
                lstY[0].append(tagAtt[i])
            else: lstY[1].append(tagAtt[i])
        # Количество true и false в первом списке
        countY0 = [0, 0]
        # Количество true и false во втором списке
        countY1 = [0, 0]
        for j in range(len(lstY[0])):
            if lstY[0][j] == 0:
                countY0[0] += 1
            else: countY0[1] += 1
        for j in range(len(lstY[1])):
            if lstY[1][j] == 0:
                countY1[0] += 1
            else: countY1[1] += 1
        if criterion == "gini":
            S1 = 1 - ((countY0[0] / len(lstY[0])) ** 2) - ((countY0[1] / len(lstY[0])) ** 2)
            S2 = 1 - ((countY1[0] / len(lstY[1])) ** 2) - ((countY1[1] / len(lstY[1])) ** 2)
            maxVal = ((len(lstY[0]) / tagAtt.size) * S1) + ((len(lstY[1]) / tagAtt.size) * S2)
        else:
            # Если хоть одна вероятность равна нулю, то и энтропия тоже равна нулю
            if countY0[0] == 0 or countY0[1] == 0:
                S1 = 0
            else:
                S1 = -(countY0[0] / len(lstY[0])) * math.log(countY0[0] / len(lstY[0]), 2) - (countY0[1] / len(lstY[0])) * math.log(countY0[1] / len(lstY[0]), 2)
            if countY1[0] == 0 or countY1[1] == 0:
                S2 = 0
            else:
                S2 = -(countY1[0] / len(lstY[1])) * math.log(countY1[0] / len(lstY[1]), 2) - (countY1[1] / len(lstY[1])) * math.log(countY1[1] / len(lstY[1]), 2)
            maxVal = 1 - ((len(lstY[0]) / tagAtt.size) * S1) - ((len(lstY[1]) / tagAtt.size) * S2)
    # Если нужна статистика критерия
    if stat == True:
        return maxVal
    # Если нужна граница
    else:
        return bound

# Класс Дерева решений
class TreeDecision:
    def __init__(self, df, tagAtt, auto=True, criterion="entropy"):
        # Данные
        self.df = df
        # Целевой атрибут
        self.tagAtt = tagAtt
        # Значения целевого атрибута
        self.tagAttVals = self.tagAtt.unique()
        # Словарь для хранения значений остальных бинарных атрибутов
        self.attsVals = {}
        # Режим построения дерева (автоматический или поэтапный)
        self.auto = auto
        # Критерий сортировки атрибутов
        self.criterion = criterion
        # Для каждого атрибута...
        for i in range(self.df.columns.size):
            # ...если значения атрибута - номинальные...
            if not isDigit(self.df.at[0, self.df.columns[i]]):
                # ...их значения сохраняются в словаре (для дальнейшего использования)
                self.attsVals[self.df.columns[i]] = self.df[self.df.columns[i]].unique()
                # ...атрибут конвертируется в бинарный
                toBin(self.df[self.df.columns[i]])
        # Конвертация целевого атрибута в бинарный
        toBin(self.tagAtt)
        # Подсчёт приоритета атрибутов
        self.sortedAtts = sortAtts(self.df.drop(self.tagAtt.name, axis=1), self.tagAtt, criterion=criterion, attsVals=self.attsVals)
        # Список для хранения блоков
        self.Blocks = {}
        # Список для хранения незавершённых ветвей
        self.freeBranches = {}
    def getFreeBranches(self):
        return self.freeBranches
    def getTagAtt(self):
        return self.tagAtt
    def getBlocks(self):
        return self.Blocks
    def getSortedAtts(self):
        return self.sortedAtts
    # Добавление ветки
    def addBranch(self, attVal, curDepth):
        # Если текущий узел не является начальным...
        if curDepth != '0':
            # ...обновляется df
            self.df = self.freeBranches[curDepth]
            # ...текущий узел перестаёт быть свободным
            del self.freeBranches[curDepth]
        # Индексы в текущем df
        indexes = self.df.index.tolist()
        # Текущий атрибут
        att = selAtt(self.sortedAtts, self.auto)
        # Удаление атрибута, чтобы его нельзя было больше использовать
        self.sortedAtts.pop(att)
        # Добавление блока в список
        self.Blocks[curDepth] = Block(self.df, self.tagAtt.name, att, self.tagAttVals, self.attsVals)
        # Разбиение объектов на 2 части + условие разбиения
        trueLst, falseLst, condition = self.Blocks[curDepth].getBound(indexes, self.auto, self.criterion)
        # Вероятность
        self.Blocks[curDepth].setP(attVal, indexes)
        # Информация о текущем блоке
        self.Blocks[curDepth].setInfo(att, condition, trueLst, falseLst, self.Blocks[curDepth].getP(), self.tagAttVals[attVal])
        # Если блок - не лист...
        if not isLeaf(self.Blocks[curDepth].getP()):
            # ...если значения объектов для trueLst - разные
            if not sameRes(self.tagAtt, trueLst):
                # ...создаётся новый узел
                self.freeBranches[curDepth + "0"] = self.df.loc[trueLst]
            # ...иначе, если значения одинаковые...
            elif len(trueLst) > 0:
                # ...создаётся лист
                self.Blocks[curDepth + "0"] = Leaf(self.tagAtt, trueLst, self.tagAtt[0], self.tagAttVals)
                p = self.Blocks[curDepth + "0"].getP()
                self.Blocks[curDepth + "0"].setInfo(len(trueLst), 0, p, self.tagAttVals[0])
            # ...если значения объектов для falseLst - разные
            if not sameRes(self.tagAtt, falseLst):  
                # ...создаётся новый узел
                self.freeBranches[curDepth + "1"] = self.df.loc[falseLst]
            # ...иначе, если значения одинаковые...
            elif len(falseLst) > 0:
                # ...создаётся лист
                self.Blocks[curDepth + "1"] = Leaf(self.tagAtt, falseLst, self.tagAtt[1], self.tagAttVals)
                p = self.Blocks[curDepth + "1"].getP()
                self.Blocks[curDepth + "1"].setInfo(0, len(falseLst), p, self.tagAttVals[1])
    def getInfo(self):
        return self.info
    # Максимальное количество узлов
    def setMaxDepth(self):
        pass

# Класс Блок
class Block:
    def __init__(self, df, tagAtt, selAtt, tagAttVals, attsVals):
        # Данные
        self.df = df
        # Выбранный атрибут
        self.selAtt = selAtt
        # Целевой атрибут
        self.tagAtt = tagAtt
        # Значения целевой переменной
        self.tagAttVals = tagAttVals
        # Словарь номинальных переменных
        self.attsVals = attsVals
    # Нахождение границ разбиения блока
    def getBound(self, indexes, auto, criterion):
        # Если атрибут содержит числа...
        try:
            self.attsVals[self.selAtt]
        except:
            # ...то он разбивается на две границы
            trueLst, falseLst, self.condition = setPartitionBoundary(self.df[self.tagAtt], self.df[self.selAtt], indexes, \
                                                                     pred=False, criterion=criterion)
            return trueLst, falseLst, self.condition
        # иначе атрибут разбивается по номинальным значениям
        else:
            trueLst, falseLst, self.condition = setAttVal(self.df[self.selAtt], indexes, self.attsVals[self.selAtt], pred=False, auto=auto)
            return trueLst, falseLst, self.condition
    def setP(self, attVal, indexes):
        count = 0
        for i in indexes:
            if self.df.at[i, self.tagAtt] == attVal:
                count += 1
        # Вероятность
        self.p = round((count / (len(indexes))), 3)
    # Подсчёт вероятности
    def getP(self):
        return self.p
    def setInfo(self, att, condition, trueLst, falseLst, p, val):
        self.val = val
        self.info = f"{att} {condition} \nCount of True = {len(trueLst)} \nCount of False = {len(falseLst)} \np = {p}" + \
            f"\nclass: {val}"
    def getInfo(self):
        return self.info
    # Прогноз по конкретному блоку
    def predict(self, df, leaf=False, pred="class", area=0.5):
        indexes = df.index.tolist()
        # Если блок не является листом...
        if not leaf:
            # ...если атрибут содержит числа...
            try:
                self.attsVals[self.selAtt]
            except:
                # ...то он разбивается на две границы
                trueLst, falseLst = setPartitionBoundary(df[self.tagAtt], df[self.selAtt], indexes, condition=self.condition[2:])
            # иначе атрибут разбивается по номинальным значениям
            else:
                trueLst, falseLst = setAttVal(df[self.selAtt], indexes, self.attsVals[self.selAtt], val=self.condition)
        # Если блок - лист...
        if isLeaf(self.p) or leaf:
            # ...если класс блока равнаяется нулевому значению...
            if self.val == self.tagAttVals[0]:
                if self.p >= area:
                    # ...если предсказывается класс...
                    if pred == "class":
                        return indexes, self.tagAttVals[0]
                    # ...если предсказывается вероятность
                    else: return indexes, self.p
                else:
                    # ...если предсказывается класс...
                    if pred == "class":
                        return indexes, self.tagAttVals[1]
                    # ...если предсказывается вероятность
                    else: return indexes, self.p
            # ...если класс блока равнаяется первому значению
            else:
                if self.p > (1 - area):
                    # ...если предсказывается класс...
                    if pred == "class":
                        return indexes, self.tagAttVals[1]
                    # ...если предсказывается вероятность
                    else: return indexes, 1 - self.p
                else:
                    # ...если предсказывается класс...
                    if pred == "class":
                        return indexes, self.tagAttVals[0]
                    # ...если предсказывается вероятность
                    else: return indexes, 1 - self.p
        # Если блок - не лист, то возвращаются списки для дальнейшего прогноза
        else: 
            return [trueLst, falseLst]
            
# Класс Лист
class Leaf:
    def __init__(self, tagAtt, indexes, attVal, tagAttVals):
        # Целевой атрибут
        self.tagAtt = tagAtt
        self.indexes = indexes
        self.attVal = attVal
        self.tagAttVals = tagAttVals
    def getP(self):
        # Если результаты всех объектов соответствуют классу
        if self.tagAtt[self.indexes[0]] == self.attVal:
            self.p = 1.0
            return self.p
        else:
            self.p = 0.0
            return self.p
    def setInfo(self, trueSize, falseSize, p, val):
        self.val = val
        self.info = f"Count of True = {trueSize} \nCount of False = {falseSize} \np = {p}" + \
            f"\nclass: {val}"
    def getInfo(self):
        return self.info
    def predict(self, df, leaf=True, pred="class", area=0.5):
        indexes = df.index.tolist()
        # ...если класс блока равнаяется нулевому значению
        if self.val == self.tagAttVals[0]:
            if self.p == 1:
                if pred == "class":
                    return indexes, self.tagAttVals[0]
                else: return indexes, self.p
            else:
                if pred == "class":
                    return indexes, self.tagAttVals[1]
                else: return indexes, self.p
        # ...если класс блока равнаяется первому значению
        else:
            if self.p == 1:
                if pred == "class":
                    return indexes, self.tagAttVals[1]
                else: return indexes, 1 - self.p
            else:
                if pred == "class":
                    return indexes, self.tagAttVals[0]
                else: return indexes, 1 - self.p


# Выбор узла
def setDepth(freeBranches, auto=True):
    if auto == True:
        # Временный узел
        tempDepth = [0, 0]
        for key in freeBranches:
            # Если количество объектов во временном узле меньше или равно количеству переменных в i-м узле...
            if tempDepth[1] <=  len(freeBranches[key][freeBranches[key].columns[0]]):
                tempDepth[1] = len(freeBranches[key][freeBranches[key].columns[0]])
                # ...то временный узел принимает значения i-го узла
                tempDepth[0] = key
        return tempDepth[0]
    else:
        print("Доступные узлы: ")
        for key in freeBranches:
            if key == list(freeBranches.keys())[-1]:
                print(key)
            else:
                print(key, end=", ")
        while True:
            depth = input("Выберите узел или введите '0' для выхода: ")
            if depth in freeBranches or depth == '0':
                return depth
            else:
                print("Такого узла нет, либо он уже использован")


# Построение дерева
def build(df, tagAtt, auto=True, maxDepth=100, criterion="entropy"):
    # Объект Дерева решений
    TreeDec = TreeDecision(df, tagAtt, auto=auto, criterion=criterion)
    # Создание первой ветки
    TreeDec.addBranch(TreeDec.getTagAtt()[0], '0')
    for depth in range(maxDepth):
        # Если нет ни свободных узлов, ни свободных атрибутов...
        if len(TreeDec.getFreeBranches()) == 0 or len(TreeDec.getSortedAtts()) == 0:
            # ...то дерево построено
            break
        # Иначе строим новую ветвь
        else:
            # Выбор узла
            curDepth = setDepth(TreeDec.getFreeBranches(), auto=auto)
            # Если "0"... 
            if curDepth == '0':
                print(TreeDec.getBlocks())
                print()
                for i in TreeDec.Blocks:
                    print(TreeDec.Blocks[i].getInfo())
                    print()
                # то заканчиваем строить дерево
                break
            # Если узел заканчивается на "0"...
            if curDepth[-1] == '0':
                # ...то класс блока принимает нулевое значение
                TreeDec.addBranch(TreeDec.getTagAtt()[0], curDepth)
            # ...иначе класс блока принимает первое значение
            else: 
                TreeDec.addBranch(TreeDec.getTagAtt()[1], curDepth)
    print(TreeDec.getBlocks())
    print()
    for i in TreeDec.Blocks:
        print(TreeDec.Blocks[i].getInfo())
        print()
    return TreeDec

# Формирование прогнозов по построенному дереву
def predict(df, TreeDec, tagAtt, pred="class", area=0.5):
    if pred == "class":
        # df до поргноза
        #print(df)
        pass
    else:
        # словарь для хранения вероятностей
        probsDict = {}
    # Словарь для хранения списков и df
    blocksBounds = {}
    # Прогноз для первого блока
    blocksBounds['0'] = TreeDec.Blocks['0'].predict(df, pred=pred, area=area)
    # Если второй элемент списка - список
    if type(blocksBounds['0'][1]) == list:
        # ...создаётся список ключей словаря,
        keys = list(TreeDec.Blocks.keys())
        # удаляется первый ключ из списка (т.к. уже зайдествован выше),
        del keys[0]
        for i in keys:
            # если ключ заканчивается на "0"...
            if i[-1] == '0':
                blocksBounds[i] = TreeDec.Blocks[i].predict(df.loc[blocksBounds[i[:-1]][0]], pred=pred, area=area)
                # ...если второй элемент списка - значение переменной (не список)...
                if type(blocksBounds[i][1]) != list:
                    # ...если прогноз по классам...
                    if pred == "class":
                        # ...замена значений df на прогнозные
                        for j in blocksBounds[i][0]:
                            df.at[j, tagAtt] = blocksBounds[i][1]
                    # ...если прогноз по вероятностям
                    else: 
                        # ...добавление вроятности в словарь
                        for j in blocksBounds[i][0]:
                            probsDict[j] = blocksBounds[i][1]
            # ...иначе если ключ заканчивается на "1"...
            else: 
                blocksBounds[i] = TreeDec.Blocks[i].predict(df.loc[blocksBounds[i[:-1]][1]], pred=pred, area=area)
                # ...если второй элемент списка - значение переменной (не список)...
                if type(blocksBounds[i][1]) != list:
                    # ...если прогноз по классам...
                    if pred == "class":
                        # ...замена значений df на прогнозные
                        for j in blocksBounds[i][0]:
                            df.at[j, tagAtt] = blocksBounds[i][1]
                    # ...если прогноз по вероятностям
                    else: 
                        # ...добавление вроятности в словарь
                        for j in blocksBounds[i][0]:
                            probsDict[j] = blocksBounds[i][1]
    # Если второй элемент списка - значение переменной (не список)...
    else:
        # ...если прогноз по классам...
        if pred == "class":
            # ...замена значений df на прогнозные
            for j in blocksBounds['0'][0]:
                df.at[j, tagAtt] = blocksBounds['0'][1]
        # ...если прогноз по вероятностям
        else: 
            # ...добавление вроятности в словарь
            for j in blocksBounds['0'][0]:
                probsDict[j] = blocksBounds['0'][1]
    # Прогноз для узлов, оставшихся свободными после построения дерева
    freeBranches = list(TreeDec.freeBranches.keys())
    for i in freeBranches:
        if i[-1] == '0':
            tmpDf = TreeDec.Blocks[i[:-1]].predict(df.loc[blocksBounds[i[:-1]][0]], leaf=True, pred=pred, area=area)
            # ...если прогноз по классам...
            if pred == "class":
                # ...замена значений df на прогнозные
                for j in tmpDf[0]:
                    df.at[j, tagAtt] = tmpDf[1]
            # ...если прогноз по вероятностям
            else: 
                # ...добавление вроятности в словарь
                for j in tmpDf[0]:
                    probsDict[j] = tmpDf[1]
        else: 
            tmpDf = TreeDec.Blocks[i[:-1]].predict(df.loc[blocksBounds[i[:-1]][1]], leaf=True, pred=pred, area=area)
            # ...если прогноз по классам...
            if pred == "class":
                # ...замена значений df на прогнозные
                for j in tmpDf[0]:
                    df.at[j, tagAtt] = tmpDf[1]
            # ...если прогноз по вероятностям
            else: 
                # ...добавление вроятности в словарь
                for j in tmpDf[0]:
                    probsDict[j] = tmpDf[1]
    if pred == "class":
        # df после поргноза
        #print(df)
        return df
    else:
        probs = [0] * len(probsDict)
        for i in range(len(probs)):
            probs[i] = probsDict[i]
        #print(probs)
        return probs

# Вывод графика ROC-кривой    
def buildROC(y, probs):
    lr_auc = roc_auc_score(y, probs)
    fpr, tpr, treshold = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
             label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Пример ROC-кривой')
    plt.legend(loc="lower right")
    plt.show()
    return(roc_auc)

# Чтение файла
df = pd.read_csv("heart.csv", sep = ";")
# Целевой атрибут
tagAtt = df['Heart']
auto = input("Введите 'a', если хотите построить дерево автоматически: ")
if auto == "a" or auto == "а":
    auto = True
else: auto = False
tests = ("gini", "pearson", "entropy")
while True:
    test = input(f"Выберите один из критериев {tests}: ")
    if test in tests:
        break
    else: print("Такого критерия нет")    
TreeDec = build(df, tagAtt, auto=auto, maxDepth=100,  criterion=test)
newDf = pd.read_csv("heart.csv", sep = ";")
probs = predict(newDf, TreeDec, tagAtt.name, pred='prob')
toBin(tagAtt, True)
roc_auc = buildROC(list(tagAtt), probs)
pred = predict(newDf, TreeDec, tagAtt.name, pred='class', area=0.5)

# =============================================================================
# lst2 = list(pred["Heart"])
# print()
# count = 0
# for i in range(len(lst1)):
#     if lst1[i] != lst2[i]:
#         count += 1
# print(count / len(lst1))
# =============================================================================




