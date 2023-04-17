import streamlit as st
import seaborn as sns
import joblib as jb
import pandas as pd
import sklearn 
st.set_page_config(
    page_title='Deployment'
)

model = jb.load('logestic.h5')

sex_enc= jb.load('sex_enc.h5')
embar_enc = jb.load('embar_enc.h5')
class_enc = jb.load('class_enc.h5')
who_enc = jb.load('who_enc.h5')
adult_male_enc = jb.load('adult_male_enc.h5')
dec_enc= jb.load('dec_enc.h5')
embark_town_enc = jb.load('embark_town_enc.h5')
alive_enc = jb.load('alive_enc.h5')
alone_enc = jb.load('alone_enc.h5')



st.write('<h1 style="text-align: center; color: offwhite;">titanic Deployment</h1>', unsafe_allow_html=True)

pclass = st.selectbox('select pclass', [3, 1, 2])
sex = st.selectbox('select gender', ['male', 'female'])
embarked = st.selectbox('select embarked', ['S', 'C', 'Q'])
clas = st.selectbox('select class', ['Third', 'First', 'Second'])
who = st.selectbox('select who was it', ['man', 'woman', 'child'])
embark_town = st.selectbox('select embark town', ['Southampton', 'Cherbourg', 'Queenstown'])

sibsp = st.radio('select sibsp', [1, 0, 3, 4, 2, 5, 8], horizontal=True)
parch = st.radio('select parch', [0, 1, 2, 5, 3, 4, 6], horizontal=True)
deck = st.radio('select deck',  ['C', 'E', 'G', 'D', 'A', 'B', 'F'],horizontal=True)
alive = st.radio('select if alive',['no', 'yes'] ,horizontal=True)

age = st.slider('choose age', 0.42, 80.0, 20.0, step=0.01)
fare = st.slider('select fare', 0.0, 512.3292, 90.56, step=0.01)

adult = st.checkbox('adult male', True)
alone = st.checkbox('alone', True)



def predict():
    df = pd.DataFrame(columns=jb.load('columns.h5')[1:])
    df.loc[0, 'pclass'] = pclass
    df.loc[0, 'sex'] = sex_enc.transform([sex])[0]
    df.loc[0, 'age'] = age
    df.loc[0, 'sibsp'] = sibsp
    df.loc[0, 'parch'] =  parch
    df.loc[0, 'fare'] = fare
    df.loc[0, 'embarked'] = embar_enc.transform([embarked])[0]
    df.loc[0, 'class'] = class_enc.transform([clas])[0]
    df.loc[0, 'who'] = who_enc.transform([who])[0]
    df.loc[0, 'adult_male'] = adult_male_enc.transform([adult])[0]
    df.loc[0, 'deck'] = dec_enc.transform([deck])[0]
    df.loc[0, 'embark_town'] = embark_town_enc.transform([embark_town])[0]
    df.loc[0, 'alive'] = alive_enc.transform([alive])[0]
    df.loc[0, 'alone'] = alone_enc.transform([alone])[0]
    
    return 'Survived' if model.predict(df)[0] == 1 else 'Died'
           
if st.button('Predict'):
        if predict() == 'Died':
            st.write(f"<h1 style= 'text-align: center; color:red;'>Died</h5>", unsafe_allow_html=True)
        else:
            st.write(f"<h1 style= 'text-align: center; '>Survived</h5>", unsafe_allow_html=True) 
# embark = st.selectbox('Select Emabrked', df['embarked'].unique())
