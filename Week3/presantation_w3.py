from typing import Dict
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pickle
import pandas as pd
from sklearn.decomposition import PCA

def addHospital(df, h, typeName):
    for row in h:
        new_row = {"X": row[0] ,"Y": row[1], "Type": typeName}
        df = df.append(new_row,ignore_index=True)
    return df
def main():
    noise = st.slider("Sigma Delta", 0.001, 10., value=2.)
    
    st.latex("Cluster1  = \mathcal{N}(0, \sigma^{2})")
    st.latex("Cluster2  = \mathcal{N}(" + f"{noise}" + ", \sigma^{2})")
    
    source_cluster1 = np.random.normal(0,0.5, (500,35)).astype(np.float32)
    target_cluster1 = np.random.normal(0,0.5, (500,160)).astype(np.float32)

    source_cluster2 = np.random.normal(0+(noise),0.5,(500,35)).astype(np.float32)
    target_cluster2 = np.random.normal(0+(noise),0.5,(500,160)).astype(np.float32)

    source_all = np.concatenate((source_cluster1,source_cluster2))
    target_all = np.concatenate((target_cluster1,target_cluster2))

    # Plot source

    pca = PCA(n_components=2)
    pca.fit(source_all)

    source_cluster1_reduced = pca.transform(source_cluster1)
    source_cluster2_reduced = pca.transform(source_cluster2)

    
    fig1 = px.scatter(x=source_cluster1_reduced[:,0],y=source_cluster1_reduced[:,1],color_discrete_sequence=['blue'])
    fig2 = px.scatter(x=source_cluster2_reduced[:,0],y=source_cluster2_reduced[:,1],color_discrete_sequence=['green'])
    
    fig = go.Figure(fig1.data + fig2.data)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Preparation")


    source_cluster1 = source_cluster1_reduced
    source_cluster2 = source_cluster2_reduced

    # 100 traning, 50 testing (In total 300)
    index = np.random.choice(np.arange(0, source_cluster1.shape[0]),300, replace=False)
    
    
    h1_source, h1_target = source_cluster1[index[0:100]], target_cluster1[index[0:100]]
    h2_source, h2_target = source_cluster1[index[100:200]], target_cluster1[index[100:200]]
    h1_test_source, h1_test_target = source_cluster1[index[200:250]], target_cluster1[index[200:250]]
    h2_test_source, h2_test_target = source_cluster1[index[250:300]], target_cluster1[index[250:300]]

    # 100 traning 50 testing (In total 150)
    index = np.random.choice(np.arange(0, source_cluster2.shape[0]),150, replace=False)
    h3_source, h3_target = source_cluster2[index[0:100]], target_cluster2[index[0:100]]
    h3_test_source, h3_test_target = source_cluster2[index[100:150]], target_cluster2[index[100:150]]
    

    data = {'X': [],
    'Y': [],
    'Type': []}
    
    df = pd.DataFrame(data)
    df = addHospital(df,h1_source, "h1_traning")
    df = addHospital(df,h2_source, "h2_traning")
    df = addHospital(df,h3_source, "h3_traning")

    df = addHospital(df,h1_test_source, "h1_testing")
    df = addHospital(df,h2_test_source, "h2_testing")
    df = addHospital(df,h3_test_source, "h3_testing")
    
    st.subheader("Fold 1")
    fig = px.scatter(df, x="X", y="Y", color="Type")
    st.plotly_chart(fig, use_container_width=True)
    #st.dataframe(df)


    # Create Fold 2

    # 100 traning, 50 testing (In total 300)
    index = np.random.choice(np.arange(0, source_cluster2.shape[0]),300, replace=False)
    
    
    h1_source, h1_target = source_cluster2[index[0:100]], target_cluster2[index[0:100]]
    h2_source, h2_target = source_cluster2[index[100:200]], target_cluster2[index[100:200]]
    h1_test_source, h1_test_target = source_cluster2[index[200:250]], target_cluster2[index[200:250]]
    h2_test_source, h2_test_target = source_cluster2[index[250:300]], target_cluster2[index[250:300]]

    # 100 traning 50 testing (In total 150)
    index = np.random.choice(np.arange(0, source_cluster1.shape[0]),150, replace=False)
    h3_source, h3_target = source_cluster1[index[0:100]], target_cluster2[index[0:100]]
    h3_test_source, h3_test_target = source_cluster1[index[100:150]], target_cluster1[index[100:150]]

    df = pd.DataFrame(data)
    df = addHospital(df,h1_source, "h1_traning")
    df = addHospital(df,h2_source, "h2_traning")
    df = addHospital(df,h3_source, "h3_traning")

    df = addHospital(df,h1_test_source, "h1_testing")
    df = addHospital(df,h2_test_source, "h2_testing")
    df = addHospital(df,h3_test_source, "h3_testing")
    
    st.subheader("Fold 2")
    fig = px.scatter(df, x="X", y="Y", color="Type")
    st.plotly_chart(fig, use_container_width=True)

def mainTitle():
    st.title("Week -3")
    st.header("by Yekta Can Tursun")
    pass
if __name__ == '__main__':
    mainTitle()
    main()




