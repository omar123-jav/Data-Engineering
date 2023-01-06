import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import math

def speedlimit(df):
    accidents_per_speed_limit = df['speed_limit'].value_counts()
    graph=px.bar(x=accidents_per_speed_limit.index, y=accidents_per_speed_limit,orientation='v')
    graph.update_layout(title="How does the speed limit of a road have an effect on the number of accidents that happened on it?",xaxis_title="Speed Limit",
    yaxis_title="No of Accidents")
    return graph
def month(df):

    accidents_per_month = df['Month'].value_counts()
    graph=px.bar(x=accidents_per_month.index, y=accidents_per_month, orientation='v')
    #thr = ['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01', '2011-05-01', '2011-06-01', '2011-07-01',
           #'2011-08-01', '2011-09-01', '2011-10-01', '2011-11-01', '2011-12-01']
    #datetimes = pd.to_datetime(thr, format='%Y-%m-%d')
    graph.update_yaxes(range=[11000, 14000])
    graph.update_layout(xaxis=dict(tickmode = 'array',tickvals=[1,2,3,4,5,6,7,8,9,10,11,12],ticktext=['January','February','March','April','May','June','July','August','September','October','November','December']), xaxis_title="Month",
    yaxis_title="No of Accidents")

    return graph
def junction(df,df2):
   junctioncount = df['junction_detail'].value_counts()
   junction_names=df2[df2['Feature']=='junction_detail']['Original Value']
   graph = px.bar(x=junctioncount.index, y=junctioncount, orientation='v')
   graph.update_layout(xaxis_title="Junction_type",yaxis_title="Number of Accidents",xaxis=dict(tickmode = 'array',tickvals=[0,1,2,3,4,5,6,7,8],ticktext=junction_names))
   return graph
def severity(df):
    df1=df.copy()

    graph=px.scatter(data_frame=df1,x='number_of_vehicles',y='number_of_casualties',color='accident_severity')
    graph.update_layout(xaxis_title="Vehicles Damaged in Accident", yaxis_title="Human Casualties in accident",xaxis=dict(tickmode = 'array',tickvals=[1,2,3,4,5,6,7]))
    return graph
def time(df):
  fig=ff.create_distplot(hist_data=[df['Hour']],group_labels=['time'],curve_type='kde',bin_size=3)
  fig.update_layout(xaxis_title="Hour", yaxis_title="Density",xaxis=dict(tickmode='array',tickvals=[0,3,6,9,12,15,18,21,23],ticktext=['12 am','3 am','6 am','9 am','12 pm','3 pm',' 6 pm','9 pm','11 pm']))
  return fig
def dashboard(filename,filename2):
    df=pd.read_csv(filename)
    df2=pd.read_csv(filename2)
    app = Dash()
    app.layout = html.Div([
        html.H1("UK_Accidents_2011 Dashboard", style={'text-align': 'center'}),
        html.Br(),
        html.Div(),
        html.H1("Relationship between speedlimit and number of accidents", style={'text-align': 'center'}),
        dcc.Graph(figure=speedlimit(df)),
        html.Br(),
        html.Div(),
        html.H1("No of accidents per month", style={'text-align': 'center'}),
        dcc.Graph(figure=month(df)),
        html.Br(),
        html.Div(),
        html.H1("Number of accidents per junction type",
                style={'text-align': 'center'}),
        dcc.Graph(figure=junction(df,df2)),
        html.Br(),
        html.Div(),
        html.H1("Effect of Vehicles Damaged in the accident on the human casualties, with respect to accident severity",
                style={'text-align': 'center'}),
        dcc.Graph(figure=severity(df)),
        html.Br(),
        html.Div(),
        html.H1("Distribution of Accidents over different times of the day",
                style={'text-align': 'center'}),
        dcc.Graph(figure=time(df)),
        html.Br(),
        html.Div()
    ])
    app.run_server(debug=False,host='0.0.0.0',port=8020)
#dashboard('../data/df_ms2.csv','../data/Lookup_Table.csv')

