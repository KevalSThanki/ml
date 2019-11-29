import numpy as np
import pandas as pd
data=pd.DataFrame(data=pd.read_csv('color.csv'))
print(data)
concepts=np.array(data.iloc[:,0:-1])
target=np.array(data.iloc[:,-1])
print(target)
def learn(concepts,target):
    specific_h=concepts[0].copy()
    for i,h in enumerate(concepts):
        if target[i]=="yes":
            for x in range(len(specific_h)):
                if h[x]!=specific_h[x]:
                    specific_h[x]="?"
    return specific_h
specific_h=learn(concepts,target)
print(specific_h)
