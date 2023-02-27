import streamlit as st
import pickle
import numpy as np
#import the model

model = pickle.load(open('svmclassi.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
scaler = pickle.load(open('StandardScaler.pkl','rb'))


st.title("Parkinson's Disease Detection")

MDVPFo = st.number_input('MDVP:Fo(Hz)')
MDVPFhi = st.number_input('MDVP:Fhi(Hz)')
MDVPFlo = st.number_input('MDVP:Flo(Hz)')
MDVPJitter = st.number_input('MDVP:Jitter(%)')
MDVPJitter = st.number_input('MDVP:Jitter(Abs)')
MDVPRAP = st.number_input('MDVP:RAP')
MDVPPPQ = st.number_input('MDVP:PPQ')
JitterDDP = st.number_input('Jitter:DDP')
MDVPShimmer = st.number_input('MDVP:Shimmer')
MDVPShimmer = st.number_input('MDVP:Shimmer(dB)')
ShimmerAPQ3 = st.number_input('Shimmer:APQ3')
ShimmerAPQ5 = st.number_input('Shimmer:APQ5')
MDVPAPQ = st.number_input('MDVP:APQ')
ShimmerDDA = st.number_input('Shimmer:DDA')
NHR = st.number_input('NHR')
HNR = st.number_input('HNR')
RPDE = st.number_input('RPDE')
DFA = st.number_input('DFA')
spread1 = st.number_input('spread1')
spread2 = st.number_input('spread2')
D2 = st.number_input('D2')
PPE = st.number_input('PPE')



if st.button('Predict result'):
    input_data = (MDVPFo,MDVPFhi,MDVPFlo,MDVPJitter,MDVPJitter,
    MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmer,ShimmerAPQ3,ShimmerAPQ5,
    MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE)
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the data
    std_data = scaler.transform(input_data_reshaped)

    prediction = model.predict(std_data)

    if prediction[0] == 0:
        st.title("The person does not have parkinson's Disease.")
    else:
        st.title("The person has parkinson's Disease.")
# 197.07600,
# 206.89600,
# 192.05500,
# 0.00289,
# 0.00001,
# 0.00166,
# 0.00168,
# 0.00498,
# 0.01098,
# 0.09700,
# 0.00563,
# 0.00680,
# 0.00802,
# 0.01689,
# 0.00339,
# 26.77500,
# 0.422229,
# 0.741367,
# -7.348300,
# 0.177551,
# 1.743867,
# 0.085569