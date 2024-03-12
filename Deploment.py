# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 20:21:01 2024

@author: hemanth_bommineni
"""


import pickle
import streamlit as st


pickle_in = open("G:\DS\Excel r\Project\decision_tree_model.pkl","rb")
classifier=pickle.load(pickle_in)


def predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk):
     prediction=classifier.predict([[industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk]])
     print(prediction)
     return prediction





def main():
    st.title("Bankruptcy Predictor")
    industrial_risk =  st.selectbox('industrial_risk', (0,0.5,1), key=1)
    management_risk =  st.selectbox('management_risk', (0,0.5,1), key=2)
    financial_flexibility =  st.selectbox('financial_flexibility', (0,0.5,1), key=3)
    credibility =  st.selectbox('credibility', (0,0.5,1), key=4)
    competitiveness =  st.selectbox('competitiveness', (0,0.5,1), key=5)
    operating_risk = st.selectbox('operating_risk', (0,0.5,1), key=6)
    result=""
    results=''
    if st.button("Predict"):
        result=predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
        if result==0:
            results='bankruptcy'
        else:
            results='non bankruptcy'
    st.success('The output is \"{}\"'.format(results))

if __name__=='__main__':
    main()