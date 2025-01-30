import streamlit as st
#st.set_option('deprecation.showPyplotGlobalUse', False)
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt

def plot_metric(label, value):
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=value,  # Ensure this is a numeric value (int or float)
            gauge={"axis": {"visible": False}},
            number={
                "valueformat": ",.1f",  # Add formatting here to show commas and decimals
                "font.size": 25,
            },
            title={
                "text": label,
                "font": {"size": 16},
            },
        )
    )

    fig.update_xaxes(visible=False, fixedrange=True)
    fig.update_yaxes(visible=False, fixedrange=True)
    fig.update_layout(
        margin=dict(t=30, b=0),
        showlegend=False,
        height=100,
    )

    st.plotly_chart(fig, use_container_width=True)

# ========== Row 1: Count plots ===========  
def employment_trend(data):
    fig = px.line(data, x='Date', y='Total_employed_persons',
                  labels={'Total_employed_persons': 'Total Employed (Thousands)',
                         'Date': 'Date'},
                  color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig, use_container_width=True)

def unemployment_trend(data):
    fig = px.line(data, x='Date', y=['unemployement_rate', 'Unemployment_rate_age_15-30'],
                  labels={'value': 'Rate (%)',
                         'Date': 'Date',
                         'variable': 'Metric'},
                  color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def age_group_analysis(data):
    # Filter rows where underemployment rates are not zero
    age_data = data[data['Underemployment_rate_by_age_15-24'] != 0]
    
    age_columns = ['Underemployment_rate_by_age_15-24',
                   'Underemployment_rate_by_age_25-34',
                   'Underemployment_rate_by_age_35-44',
                   'Underemployment_rate_by_age_45+']
    
    fig = px.line(age_data, x='Date', y=age_columns,
                  labels={'value': 'Underemployment Rate (%)',
                         'Date': 'Date',
                         'variable': 'Age Group'},
                  color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def participation_trend(data):
    fig = px.line(data, x='Date', y=['Labor_force_participation_rate', 'Employment_Population_ratio_rate'],
                  labels={'value': 'Rate (%)',
                         'Date': 'Date',
                         'variable': 'Metric'},
                  color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def monthly_changes(data):
    # Calculate month-over-month changes
    data['Employment_Change'] = data['Total_employed_persons'].diff()
    
    fig = px.bar(data, x='Date', y='Employment_Change',
                 labels={'Employment_Change': 'Monthly Change in Employment (Thousands)',
                        'Date': 'Date'},
                 color='Employment_Change',
                 color_continuous_scale='RdBu')
    st.plotly_chart(fig, use_container_width=True)