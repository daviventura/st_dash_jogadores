import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(
    layout='wide',
    page_title='Mercado Jogadores'
)

col1_1,col2_2=st.columns([0.2,0.8])
col1,col2=st.columns([0.6,0.4])

df=pd.read_excel('datasets/dados_inputs.xlsx')
paises_origem=[i.replace('Primeira Nacionalidade_','') for i in df.columns[1:21]]
paises_destino=[i.replace('Destino_','') for i in df.columns[21:27]]


# VARIÁVEIS

x_sm=df[df.columns[:21]]
y_sm=df[['Classe País']]
x_lr=df[df.columns[:27]]
y_lr=df[['Ln VM']]

x_train,x_test,y_train,y_test=train_test_split(x_lr,y_lr,test_size=0.3)


# MODELOS

sm=LogisticRegression(multi_class='multinomial')
sm.fit(x_sm,y_sm)

lr=LinearRegression()
lr.fit(x_lr,y_lr)

x=lr.predict(x_lr)
y=y_lr

# PREVISÕES - PROBABILIDADES

col1_1.image('imagens/Logo PNG.png',width=250)
pais_origem=col1_1.selectbox('País de Origem',paises_origem)
idade=col1_1.select_slider('Idade',list(range(12,40)))

df_sm=pd.DataFrame({i:[0] for i in df.columns[:27]})
df_sm.loc[0,'Idade']=idade
df_sm.loc[0,'Primeira Nacionalidade_'+pais_origem]=1

prob_pais=sm.predict_proba(df_sm[df_sm.columns[:21]])
lista_prob=[float('{:.0f}'.format(prob_pais[0,i]*100)) for i in range(6)]

df_bar_sm=pd.DataFrame({'Probabilidade':lista_prob},index=paises_destino)


# PREVISÕES - VALOR DE MERCADO

df_lr=pd.DataFrame({i:[0] for i in df.columns[:27]})

for i in range(len(paises_destino)):
    df_lr.loc[i,'Idade']=idade
    df_lr.loc[i,'Primeira Nacionalidade_'+pais_origem]=1
    df_lr.loc[i,'Destino_'+paises_destino[i]]=1
df_lr.fillna(0,inplace=True)

vm_previsto=[float('{:.1f}'.format(np.exp(lr.predict(df_lr)[i,0]),)) for i in range(len(paises_destino))]
df_bar_lr=pd.DataFrame({'VM Previsto':vm_previsto},index=paises_destino)


# MAPA

latitude=[40.41678,48.85661,51.50735,41.90278,38.71667,39.93336]
longitude=[-3.70379,2.35222,-0.12776,12.49636,-9.139,32.85974]
size=vm_previsto

df_map=pd.DataFrame({'Latitude':latitude,
                     'Longitude':longitude,
                     'Size':size})
df_map['Size']=df_map['Size']*3500

maior_vm_pais=df_bar_lr.sort_values(by='VM Previsto',ascending=False).index[0]

col2_2.header('VALOR DE MERCADO MUNDIAL')
col2_2.metric('Maior Valor de Mercado',
            value=str(max(vm_previsto))+' Milhões € - '+maior_vm_pais)

with col2_2.container(height=300,border=False):
    col2_2.map(df_map,latitude='Latitude',longitude='Longitude',size='Size',use_container_width=False)


# GRÁFICOS

col2_2.subheader('Valor de Mercado Previsto - Em € Milhões')
col2_2.bar_chart(df_bar_lr)

col2_2.subheader('Probabilidade Venda / País (%)')
col2_2.bar_chart(df_bar_sm)
