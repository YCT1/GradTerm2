import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
address = "https://raw.githubusercontent.com/YCT1/GradTerm2/master/Week4/"
#address = ""
def locallyTrained(fold):
    
    df = pd.DataFrame()
    hospital_number = 3
    for i in range(1, hospital_number+1):
        if i == hospital_number:
            df[f"Local Hospital {i} OOD"] = pd.read_csv(f"{address}Results/Sel1_fold{fold}/h_{i}_1_loss.csv")
        else:
            df[f"Local Hospital {i}"] = pd.read_csv(f"{address}Results/Sel1_fold{fold}/h_{i}_1_loss.csv")


    fig = px.line(df, y=df.columns,title=f"Localy Trained Model 500 Epochs Fold {fold}")
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="MAE",
        legend_title="Hospitals",
    )

    st.plotly_chart(fig, use_container_width=True)

def traditonalFedL(fold, id=2, title="FedL Model"):
    df = pd.DataFrame()
    hospital_number = 3
    cycle = 5 


    for i in range(1, hospital_number+1):
        hospital_loss_values = list()
        for k in range(1, cycle + 1):
            loss_values = pd.read_csv(f"{address}Results/Sel{id}_fold{fold}/h_{i}_{k}_loss.csv").values
            hospital_loss_values.append(loss_values)
        hospital_loss_values = np.array(hospital_loss_values).flatten()
        if i == hospital_number:
            df[f"FedL Hospital {i} OOD"] = hospital_loss_values
        else:
            df[f"FedL Hospital {i}"] = hospital_loss_values
        
    fig = px.line(df, y=df.columns,title=f"{title} 5 Cycle with 100 Epochs each Fold {fold}")
    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="MAE",
        legend_title="Hospitals",
    )
    st.plotly_chart(fig, use_container_width=True)

def addHospital(df, h, typeName):
    for row in h:
        new_row = {"X": row[0] ,"Y": row[1], "Type": typeName}
        df = df.append(new_row,ignore_index=True)
    return df
def mainweek3():
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

def main():
    st.title("Week 4")
    st.text("Yekta Can Tursun")

    st.header("How We created the simulated data")
    mainweek3()

    st.header("Locally Trained")
    st.subheader("500 Epochs")
    locallyTrained(1)
    locallyTrained(2)

    st.header("FedL Trained")
    st.subheader("5 Cycles with 100 Epochs each")
    traditonalFedL(1)
    traditonalFedL(2)

    st.header("FedL Trained with Alignment")
    st.subheader("5 Cycles with 100 Epochs each, Top 5 layer")
    traditonalFedL(1, 3,"FedL Model with Aligment K=5, ")
    traditonalFedL(2, 3,"FedL Model with Aligment K=5, ")


    st.subheader("5 Cycles with 100 Epochs each, Top 10 layer")
    traditonalFedL(1, 4,"FedL Model with Aligment K=10, ")
    traditonalFedL(2, 4,"FedL Model with Aligment K=10, ")

    st.header("FedL Trained with Alignment Bi-Directional")
    st.subheader("5 Cycles with 100 Epochs each, Top 5 layer")
    traditonalFedL(1, 5,"FedL Model with Aligment K=5, ")
    traditonalFedL(2, 5,"FedL Model with Aligment K=5, ")


    st.subheader("5 Cycles with 100 Epochs each, Top 10 layer")
    traditonalFedL(1, 6,"FedL Model with Aligment K=10, ")
    traditonalFedL(2, 6,"FedL Model with Aligment K=10, ")
    
    

if __name__ == '__main__':
    main()