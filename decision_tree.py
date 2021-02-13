import numpy as np
import pandas as pd
import math


class node:

    def __init__(self,leave=True,pos=None,feature_name = None,class_type=None):


        self.leave = leave
        self.chlid = {}
        self.feature_pos = pos
        self.feature_name=feature_name
        self.class_type =class_type

    #case:list
    def predict(self,case):

        if self.leave == True:

            return self.class_type

        return self.chlid[case[self.feature_pos-1]].predict(case)

            


class decision_tree:


    def __init__(self, base = 2,epsilon = 0.2):


        self.data = pd.read_excel("Data.xlsx",engine="openpyxl",index=False)
        self.base = base
        #self.feature_list = [ i for i in self.data.columns[1:] ]
        self.epsilon = epsilon
        self.tree_root = self.construct_tree(self.data)


    def empirical_entropy(self,data):

        entropy = 0

        D = data.shape[0]

        for i in data.iloc[:,0].unique():

            Ck = data.loc[data[data.columns[0]]==i].shape[0]

            p = Ck/D

            entropy += p*math.log(p,self.base)

        return (-entropy)


    def empirical_conditional_entropy(self,data,dim):

        D = data.shape[0]

        cond_entropy = 0

        for i in data.iloc[:,dim].unique():

            Di = data.loc[data[data.columns[dim]]==i]

            HD = self.empirical_entropy(Di)

            cond_entropy += (Di.shape[0]/D)*HD

        return cond_entropy


        
    def mutual_information(self,ent,cond_ent):

        return (ent - cond_ent)

    def information_gain_ratio(self,ent,cond_ent,data,dim):

        f_value_list = data.iloc[:,dim].unique()

        D = data.shape[0]

        HAD = 0

        for i in f_value_list:

            p = (data.loc[data[data.columns[dim]]==i].shape[0])/D

            HAD += (-p*math.log(p,self.base))

        print(HAD)

        return ((self.mutual_information(ent,cond_ent))/HAD)


    def Get_Best_Feature(self,data,algo = "ID3"):

        HD = self.empirical_entropy(data)

        ent_list = []

        for i in range(1,data.shape[1]):

            HDA = self.empirical_conditional_entropy(data,i)

            if algo == "ID3":

                ent_list.append((i,data.columns[i],self.mutual_information(HD,HDA)))

            elif algo == "C4.5":

                ent_list.append((i,data.columns[i],self.information_gain_ratio(HD,HDA,data,i)))

        return max(ent_list,key = lambda x:x[-1])
    

    def construct_tree(self,data,algo="ID3"):


        class_data = data.iloc[:,0].unique()
        feature_list = data.columns[1:]

        if class_data.size == 1:

            return node(leave=True,class_type=class_data[0])


        if len(feature_list)==0:

            return node(leave=True,class_type=data.iloc[:,0].value_counts().sort_values(ascending=False).index[0])


        max_feature_pos,max_feature_name,max_feature_ent = self.Get_Best_Feature(data,algo)

        if max_feature_ent < self.epsilon:

            return node(leave=True,class_type=data.iloc[:,0].value_counts().sort_values(ascending=False).index[0])

        
        Ag_list = data[max_feature_name].unique()

        node_tree = node(leave=False,pos=max_feature_pos,feature_name=max_feature_name)

        for f_value in Ag_list:

            sub_tree_data = data.loc[data[max_feature_name]==f_value].drop([max_feature_name],axis=1)

            node_tree.chlid[f_value] = self.construct_tree(sub_tree_data)


        return node_tree



    #Case:List
    def predict(self,case):
        
        return self.tree_root.predict(case)
