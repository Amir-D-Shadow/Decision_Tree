import decision_tree as dt
import math
import pandas as pd
import numpy as np

a = dt.decision_tree()

b = a.empirical_entropy(a.data)

c1 = a.empirical_conditional_entropy(a.data,1)

c2 = a.empirical_conditional_entropy(a.data,2)

c3 = a.empirical_conditional_entropy(a.data,3)

c4 = a.empirical_conditional_entropy(a.data,4)

d1 =  a.mutual_information(b,a.empirical_conditional_entropy(a.data,1))

d2 = a.mutual_information(b,a.empirical_conditional_entropy(a.data,2))

d3 = a.mutual_information(b,a.empirical_conditional_entropy(a.data,3))

d4 = a.mutual_information(b,a.empirical_conditional_entropy(a.data,4))

e = a.empirical_entropy(a.data.loc[a.data[a.data.columns[3]]=="No"])

f1 = a.empirical_conditional_entropy(a.data.loc[a.data[a.data.columns[3]]=="No"],1)

f2 = a.empirical_conditional_entropy(a.data.loc[a.data[a.data.columns[3]]=="No"],2)

f3 = a.empirical_conditional_entropy(a.data.loc[a.data[a.data.columns[3]]=="No"],4)

g1 = a.mutual_information(e,f1)

g2 = a.mutual_information(e,f2)

g3 = a.mutual_information(e,f3)

j,h,i = a.Get_Best_Feature(a.data)

j = a.Get_Best_Feature(a.data)

k = a.predict([3, "No", "No", "Excellent"])

































