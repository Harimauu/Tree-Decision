import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# Для графика дерева решений
import pygraphviz as pg
# ROC-кривая
from sklearn.metrics import roc_curve, auc
# Критерий Хи-квадрат
from scipy.stats import chi2_contingency


# Класс Дерево Решений (ДР)
class DecisionTree:
    def __init__(self, df, tar_att, tar_val, mode='auto', criterion='entropy', max_branches=10, min_objs=1):
        # DataFrame
        if isinstance(df, pd.DataFrame):
            self.df = df
        else:
            raise Exception(f"{df} must belong to the DataFrame class")
        # Целевой атрибут
        if tar_att in df.columns:
            if len(self.df[tar_att].unique()) == 2:
                self.tar_att = tar_att
            else: 
                raise Exception("Target attribute must contain 2 values")
        else:
            raise Exception(f"Column {tar_att} is missing from DataFrame")
        # Целевое значение
        if tar_val in tuple(df[tar_att]):
            self.tar_val = tar_val
        else:
            raise Exception(f"Value {tar_val} is missing from {tar_att}")
        # Режим построения (автоматический/поэтапный)
        self.mode = mode if mode == 'phased' else 'auto'
        # Критерий для построения ДР (энтропии/Джини/хи-квадрат)
        if criterion == 'gini':
            self.criterion = criterion
        elif criterion == 'chi2':
            self.criterion = criterion
        else: self.criterion = 'entropy'
        # Максимальное количество (колво) ветвей в длину
        self.max_branches = max_branches if isinstance(max_branches, int) and max_branches > 0 else 10
        # Минимальное колво объектов в ветке
        self.min_objs = min_objs if isinstance(min_objs, int) and min_objs > 0 else 1
        # Качественные переменные
        self.qual_atts = []
        # Количественные переменные
        self.quant_atts = []
        for i in self.df.columns:
            if self.df[i].dtype == 'object':
                self.qual_atts.append(i)
            else: self.quant_atts.append(i)
        self.qual_atts.remove(self.tar_att)
        # Ветки
        self.branches = {'0': Branch(self.df, self.tar_att, self.tar_val, self.mode, self.criterion, self.qual_atts, self.quant_atts)}
        self.branches['0'].sort_atts(self.qual_atts + self.quant_atts)
        # Свободные узлы
        self.free_nodes = ['00', '01']
    
    def build(self):
        while len(self.free_nodes) > 0:
            DecisionTree.add_branch(self)
        
    # Добавить ветку
    def add_branch(self):
        # Выбор узла
        node = DecisionTree.__select_node(self)
        # При выходе из поэтапного режима построения дерева
        if node == "-1":
            for i in self.free_nodes:
                df = self.branches[i[:-1]].df.loc[self.branches[i[:-1]].get_indexes()]
                self.branches[i] = Leaf(df, self.tar_att, self.tar_val, self.mode, self.criterion, self.qual_atts, self.quant_atts)
            self.free_nodes = []
        else:
            # Обновление df
            df = self.branches[node[:-1]].df.loc[self.branches[node[:-1]].get_indexes()]
            # Если текущая ветка не лист...
            if df[self.tar_att][df[self.tar_att] == self.tar_val].size != df.shape[0] and df[self.tar_att][df[self.tar_att] == self.tar_val].size != 0:
                # Добавление ветки
                self.branches[node] = Branch(df, self.tar_att, self.tar_val, self.mode, self.criterion, self.qual_atts, self.quant_atts)
                # Целевой атрибут + уже использованные атрибуты
                del_columns = {self.tar_att}
                # Множество оставшихся атрибутов
                temp_node = node[:-1]
                while len(temp_node) > 0:
                    del_columns.add(self.branches[temp_node].sel_att)
                    temp_node = temp_node[:-1]
                columns = set(self.df.columns) - del_columns
                # Выбор атрибута
                temp = self.branches[node].sort_atts(columns)
                # Если дерево
                if (len(node) <= (self.max_branches + 1)) and (df.shape[0] >= self.min_objs) and (temp != False):
                    self.free_nodes.append(node + '0')
                    self.free_nodes.append(node + '1')
                else: 
                    self.branches[node] = Leaf(df, self.tar_att, self.tar_val, self.mode, self.criterion, self.qual_atts, self.quant_atts)
            # Если текущая ветка лист
            else:
                self.branches[node] = Leaf(df, self.tar_att, self.tar_val, self.mode, self.criterion, self.qual_atts, self.quant_atts)
            self.free_nodes.remove(node)
        
    # Выбор узла
    def __select_node(self):
        # При поэтапном построении...
        if self.mode == 'phased':
            print("Доступные узлы: ")
            for i in self.free_nodes:
                if i == self.free_nodes[-1]:
                    print(i, end='')
                else:
                    print(i, end=", ")
            while True:
                node = input("Выберите ветку или введите '-1' для выхода: ")
                if node in self.free_nodes or node == '-1':
                    print()
                    break
                else:
                    print("Такой ветки нет, либо она уже использована", end='')
        # При автоматическом построении...
        else:
            # Наибольший размер ветки-родителя
            max_size = 0
            # Перебор всех свободных узлов
            for i in self.free_nodes:
                # - если узел оканчивается на "0"
                if i[-1] == '0':
                    # -- если есть противоположный узел
                    if (i[:-1] + '1') in self.branches:
                        size = self.branches[i[:-1]].df.shape[0] - self.branches[(i[:-1] + '1')].df.shape[0]
                        temp_node = i
                    # -- если нет противоположного узла
                    else:
                        # --- если индексов True больше, чем индексов False
                        if len(self.branches[i[:-1]].unsorted_indexes[0]) >= len(self.branches[i[:-1]].unsorted_indexes[1]):
                            # ---- берётся текущий узел
                            temp_node = i
                        # --- иначе берётся противоположный узел
                        else: temp_node = i[:-1] + '1'
                        size = self.branches[i[:-1]].df.shape[0]
                # - если узел заканчивается на '1'
                else:
                    if (i[:-1] + '0') in self.branches:
                        size = self.branches[i[:-1]].df.shape[0] - self.branches[(i[:-1] + '0')].df.shape[0]
                        temp_node = i
                    else: 
                        continue
                if size > max_size:
                    max_size = size
                    node = temp_node
        return node
    
    # Нарисовать дерево
    def draw(self):
        A = pg.AGraph(directed=True)
        # Первая ветка
        A.add_node('0', color='green', style='filled', label=self.branches['0'].get_info(), shape='box')
        keys = list(self.branches.keys())
        del keys[0]
        # Остальные ветки
        for i in range(len(keys)):
            if isinstance(self.branches[keys[i]], Leaf):
                if keys[i][-1] == '0':
                    cr = 'aqua'
                else: 
                    cr = 'red'
            else:
                if keys[i][-1] == '0':
                    cr = 'darkorchid1'
                else: 
                    cr = 'coral1'
            A.add_node(keys[i], color=cr, style='filled', label=self.branches[keys[i]].get_info(), shape='box')
            # Если ID узлов не "00" и не "01"...
            if len(keys[i]) != 2:
                # - то узлы соединяются с предыдущими узлами
                A.add_edge(keys[i][:-1], keys[i], label=keys[i])
            else:
                # - иначе узлы соединяются с узлом "0"
                A.add_edge('0', keys[i], label=keys[i])
        # Сохранение графика
        A.layout()
        A.draw("TreeDec.png", prog='dot')
        
    # Получить вероятности
    def get_probs(self, df):
        probs = [0] * df.shape[0]
        # Если вероятности получаются для изначального DF
        if id(df) == id(self.df):
            # - когда узел является листом, объекты, находящиеся в этом листе, получают его вероятность
            for key in self.branches:
                if isinstance(self.branches[key], Leaf):
                    indexes = self.branches[key].df.index.tolist()
                    for i in indexes:
                        probs[i] = self.branches[key].p
        # Иначе
        else:
            # - перебор всех объектов
            for i in range(df.shape[0]):
                key = '0'
                # - перебор всех узлов, пока объект не попадёт в лист
                while not isinstance(self.branches[key], Leaf):
                    if isinstance(self.branches[key].cond, str):
                        if df.loc[i, self.branches[key].sel_att] == self.branches[key].cond:
                            key += '0'
                        else: key += '1'
                    else:
                        if df.loc[i, self.branches[key].sel_att] < self.branches[key].cond:
                            key += '0'
                        else: key += '1'
                probs[i] = self.branches[key].p
        return probs
    
    # Построение ROC-кривой
    def build_roc(self, y_vals, probs):
        # Замена значений y_vals на 1 и -1
        for i in range(len(y_vals)):
            y_vals[i] = 1 if y_vals[i] == self.tar_val else -1
        # Нахождение значений ROC
        fpr, tpr, treshold = roc_curve(y_vals, probs)
        dist = []
        for i in range(len(fpr)):
            dist.append(tpr[i] - fpr[i])
        roc_auc = auc(fpr, tpr)
        values = (round(roc_auc, 3), round(fpr[dist.index(max(dist))], 2), round(tpr[dist.index(max(dist))], 2))
        # Построение графика
        plt.plot(fpr, tpr, color='darkorange',
                 label='ROC кривая (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривоя')
        plt.legend(loc="lower right")
        plt.show()
        return values
    

# Класс Ветка
class Branch:
    def __init__(self, df, tar_att, tar_val, mode, criterion, qual_atts, quant_atts):
        self.df = df
        self.tar_att = tar_att
        self.tar_val = tar_val
        self.mode = mode
        self.criterion = criterion
        self.qual_atts = qual_atts
        self.quant_atts = quant_atts
        # Вероятность
        self.p = round(self.df[self.tar_att][self.df[self.tar_att] == self.tar_val].size / self.df.shape[0], 2)
        # Уникальные значения целевого атрибута
        self.tar_att_vals = self.df[self.tar_att].unique()
        # Выбранный атрибут
        self.sel_att = ''
        # Условие для атрибута
        self.cond = ''
        # Список для хранения True и False объектов
        self.indexes = [0, 0]
        self.unsorted_indexes = []
    
    # Перегрузка оператора print при выводе словаря
    def __repr__(self):
        return f'{self.df}'
    
    # Сортировка атрибутов
    def sort_atts(self, columns):
        # Получение статистики значимости атрибутов
        sorted_atts = {}
        for i in columns:
            if Branch.calc_of_stat(self, i)[0] > 0:
                sorted_atts[i] = Branch.calc_of_stat(self, i)
        # Сортировка атрибутов по убыванию
        sorted_atts = sorted(sorted_atts.items(), key=lambda item: item[1][0], reverse=True)
        sorted_atts = {k: v for k, v in sorted_atts}
        if len(sorted_atts) == 0:
            return False
        # Выбор атрибута
        # - автоматический режим
        if self.mode == 'auto':
            self.sel_att = tuple(sorted_atts.keys())[0]
            self.cond = sorted_atts[self.sel_att][1]
        # - поэтапный режим
        else:
            print(f'Доступные атрибуты: {sorted_atts}', end='')
            while True:
                self.sel_att = input("Выберите атрибут: ")
                if self.sel_att in sorted_atts:
                    self.cond = sorted_atts[self.sel_att][1]
                    print()
                    break
                else:
                    print("Такого атрибута нет", end='')
        # Разбиение df на две части
        if self.sel_att in self.qual_atts:
            self.indexes[0] = self.df[self.sel_att][self.df[self.sel_att] == self.cond].index.tolist()
            self.indexes[1] = self.df[self.sel_att][self.df[self.sel_att] != self.cond].index.tolist()
        else:
            self.indexes[0] = self.df[self.sel_att][self.df[self.sel_att] < self.cond].index.tolist()
            self.indexes[1] = self.df[self.sel_att][self.df[self.sel_att] >= self.cond].index.tolist()
        # Конечное значение следующего узла
        self.unsorted_indexes = self.indexes.copy()
        self.indexes.sort(key=lambda i: len(i), reverse=True)
    
        
    # Расчёт статистики по критериям
    def calc_of_stat(self, att):
        # Список уникальных значений выбранного атрибута
        att_vals = self.df[att].unique()
        # Если атрибут количественный, то выполняется сортировка его элементов
        if att in self.quant_atts:
            att_vals.sort()
        # Максимальное значение статистики
        max_stat = 0
        # Условие
        cond = 0
        # Количество уникальных значений целевой переменной
        count = len(self.tar_att_vals)
        # Если критерий хи-квадрат...
        if self.criterion == 'chi2':
            # Перебор всех значений выбранного атрибута
            for i in range(1, len(att_vals)):
                # Обновление df
                new_df = self.df.copy()
                # Если атрибут качественный...
                if att in self.qual_atts:
                    new_df.loc[new_df[att] == att_vals[i], att] = f'{att_vals[i]}'
                    new_df.loc[new_df[att] != att_vals[i], att] = '-1'
                # Иначе...
                else:
                    # Временное условие
                    new_df.loc[new_df[att] < att_vals[i], att] = att_vals[i]-1
                    new_df.loc[new_df[att] >= att_vals[i], att] = att_vals[i]
                # Расчёт значения статистики
                stat = Branch.calc_of_chi2(new_df[att], new_df[self.tar_att])
                if max_stat < stat:
                    max_stat = stat
                    cond = att_vals[i]
        # Иначе...
        else:
            # Перебор всех значений выбранного атрибута
            for i in range(1, len(att_vals)):
                # - список для хранения поделённого на две части df
                x = [0, 0]
                # - если атрибут качественный...
                if att in self.qual_atts:
                    x[0] = self.df.loc[self.df[att] == att_vals[i]]
                    x[1] = self.df.loc[self.df[att] != att_vals[i]]
                # - иначе...
                else:
                    x[0] = self.df.loc[self.df[att] < att_vals[i]]
                    x[1] = self.df.loc[self.df[att] >= att_vals[i]]
                S = [0, 0]
                for j in range(2):
                    # -- количество элементов с целевым значением
                    y = [0] * count
                    for k in range(count):
                        y[k] = x[j][self.tar_att][x[j][self.tar_att] == self.tar_att_vals[k]].size
                        # --- если критерий энтропии
                        if self.criterion == 'entropy':
                            # ---- если элементы с определённым значением отсутствует, то пропуск итерации (их энтропия равна нулю)
                            if y[k] == 0:
                                continue
                            S[j] += Branch.calc_of_entropy(x[j], y[k])
                        # --- если критерий Джини
                        if self.criterion == 'gini':
                            S[j] = 1
                            S[j] -= Branch.calc_of_gini(x[j], y[k])
                # - расчёт значения статистики
                stat = 1
                for j in range(2):
                    stat -= (x[j].shape[0] / self.df.shape[0]) * S[j]
                if max_stat < stat:
                    max_stat = stat
                    cond = att_vals[i]
                # -- если у выбранного атрибуте всего 2 значения, то выход из цикла (т.к. смысла менять местами нет)
                if len(att_vals) == 2:
                    break
        return (max_stat, cond)
    
    # Расчёт статистики по критерию энтропии
    @staticmethod
    def calc_of_entropy(x, y): 
        return -((y/x.shape[0]) * np.log2(y/x.shape[0]))
    
    @staticmethod            
    # Расчёт статистики по критерию Джини
    def calc_of_gini(x, y):
        return ((y/x.shape[0]) ** 2)
    
    @staticmethod
    # Расчёт статистики по критерию хи-квадрат
    def calc_of_chi2(x, y):
        table = pd.crosstab(x, y)
        stat = chi2_contingency(table)
        return stat[0]
       
    # Геттер индексов для следующей ветки
    def get_indexes(self):
        # При первом вызове первая дочерняя ветка получает список индексов с наибольшим 
        # количеством объектов, а при повторном вызове вторая дочерняя ветка получает 
        # список с оставшимися объектами
        temp_inds = self.indexes[0]
        del self.indexes[0]
        return temp_inds
    
    def get_info(self):
        sign = 'is' if self.sel_att in self.qual_atts else '<'
        info = f'{self.sel_att} {sign} {self.cond}\nCount of True: {len(self.unsorted_indexes[0])}\n\
Count of False: {len(self.unsorted_indexes[1])}\n\nP of "{self.tar_val}" = {self.p}' 
        return info


# Класс Лист
class Leaf(Branch):
    def __init__(self, *args):
        super().__init__(*args)
    
    def get_info(self):
        info = f'Count: {self.df.shape[0]}\nP of "{self.tar_val}" = {self.p}'
        return info


if __name__ == "__main__":
    df = pd.read_csv("heart.csv", sep = ";")
    tar_val = 'Heart'
    dec_tree = DecisionTree(df, tar_val, 'yes', 'auto', 'gini', 10, 1)
    dec_tree.build()
    dec_tree.draw()
    y_vals = df[tar_val].tolist()
    probs = dec_tree.get_probs(df)
    values = dec_tree.build_roc(y_vals, probs)
    print(f"AUC: {values[0]}")
    
