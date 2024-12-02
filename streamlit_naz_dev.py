import streamlit as st
import pandas as pd # version 2.0.3
import numpy as np
import altair as alt # version 4.2.2
import seaborn as sns # version 0.13.1
import plotly.express as px # version 5.15.0
import matplotlib.pyplot as plt # version 3.7.1
import matplotlib.ticker as mtick


############################
# load files for dashboard #
############################
totals = pd.read_csv('//Users//nuremek//Documents//MADS//SIADS699//totals.csv')
df_imp_evnt_agg = pd.read_csv('//Users//nuremek//Documents//MADS//SIADS699//install_event_binned_users.csv')
partners = pd.read_csv('//Users//nuremek//Documents//MADS//SIADS699//marketing_partners_users.csv')
states = pd.read_csv('//Users//nuremek//Documents//MADS//SIADS699//states_totals.csv')


#########################################
# [Visual 1] Users by Marketing Partner #
#########################################
fig1, ax1 = plt.subplots(figsize=(6, 4))
fig1.suptitle("Total Users by by Marketing Partner", fontsize=14, y=0.95)

bars = plt.bar(partners['marketing_partner'], partners['device_id'], color='#00274C') # '#00274C','#FFCB05'
for bar in bars:
    yval = bar.get_height()
    plt.annotate(f'{int(yval)}',
                 xy=(bar.get_x() + bar.get_width() / 2, yval),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords='offset points',
                 ha='center', va='bottom')

# plot average
plt.axhline(y=partners['device_id'].mean(), color='#FFCB05', ls='--', label='Average') # '#00274C','#FFCB05'

plt.ylabel('Total Users',fontsize=10)
plt.xlabel('Days', fontsize=10)

# Set the y limits making the maximum 10% greater
ymin, ymax = min(partners['device_id']), max(partners['device_id'])
plt.ylim(ymin, 1.1 * ymax)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.xticks(rotation=45)
plt.legend()


###########################
# OR display as pie chart #
###########################
labels = partners['marketing_partner']
sizes = partners['device_id']

total_device_id = partners['device_id'].sum()
partners['percentage'] = partners['device_id'] / total_device_id * 100

# define color theme for chart
color_theme = px.colors.qualitative.Set3

# Create the Pie Chart
fig4 = px.pie(
    partners,
    names='marketing_partner',      # Labels
    values='device_id',             # Values
    title='Device Distribution by Marketing Partner',
    hover_data={'device_id': True, 'percentage': True},  # Hover details
    labels={'device_id': 'Total Users', 'percentage': 'Percentage'},
    color_discrete_sequence=color_theme  # Apply the color theme
)

# Customize hover template to display total device_id and percentage
fig4.update_traces(
    textinfo='percent',            # Show percentages on the pie chart
    hovertemplate='<b>%{label}</b><br>Total Devices: %{value}<br>Percentage: %{percent}'
)


##############################################
# [Visual 2] Days from Campaign to Streaming #
##############################################
# plot each attribute into its own bar chart
fig2, ax2 = plt.subplots(figsize=(6, 4))
fig2.suptitle("Days from Campaign to Streaming", fontsize=14, y=0.95)

bars = plt.bar(df_imp_evnt_agg['imp_evnt_binned'], df_imp_evnt_agg['device_id'], color='#00274C') # '#00274C','#FFCB05'
for bar in bars:
    yval = bar.get_height()
    plt.annotate(f'{int(yval/1000)}K',
                 xy=(bar.get_x() + bar.get_width() / 2, yval),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords='offset points',
                 ha='center', va='bottom')

# plot average
plt.axhline(y=df_imp_evnt_agg['device_id'].mean(), color='#FFCB05', ls='--', label='Average') # '#00274C','#FFCB05'

plt.ylabel('Total Users',fontsize=10)
plt.xlabel('Days', fontsize=10)

# Set the y limits making the maximum 10% greater
ymin, ymax = min(df_imp_evnt_agg['device_id']), max(df_imp_evnt_agg['device_id'])
plt.ylim(ymin, 1.1 * ymax)
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.xticks(rotation=45)
plt.legend()


##############################################################
# [Visual 3] Total Users or Average Minutes Watched by State #
##############################################################
import plotly.express as px
import json

fig3 = px.choropleth(
    states,
    locations='state',             # State abbreviations
    locationmode='USA-states',     # Map U.S. states
    color='users',     # Metric for color intensity
    color_continuous_scale='Viridis',
    hover_name='state',            # Hover label
    hover_data=['users','minutes','impressions','clicks','minutes_per_user'],  # Additional info
    title='Totals by State',
    scope='usa'                    # Focus on USA
)


####################
# Deploy streamlit #
####################
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Dashboard `version 2`')

st.sidebar.subheader('Choose your marketing partner')
time_hist_color = st.sidebar.selectbox('Color by', ('marketing_partner')) 

st.sidebar.subheader('Choose your campaign')
donut_theta = st.sidebar.selectbox('Select data', ('creative_name'))

st.sidebar.subheader('Campaign Dates')
# plot_data = st.sidebar.multiselect('Select data', ['parter1', 'partner1'], ['campaign1', 'campaign2'])
# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by Plutonians.
''')

# Row A

def format_number(num):
    if num > 1000000000:
        if not num % 1000000000:
            return f'{num // 1000000000} B'
        return f'{round(num / 1000000000, 1)} B'
    
    elif num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    
    return f'{num // 1000} K'

# top four metrics
devices = format_number(totals['total_users'][0])
minutes = format_number(totals['total_minutes'][0])
impressions = format_number(totals['total_impressions'][0])
clicks = format_number(totals['total_clicks'][0])

st.markdown('### Metrics')
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Users", devices) #, "1.2 °F")
col2.metric("Total Minutes Watched", minutes) #, "-8%")
col3.metric("Total Ad Impressions", impressions) #, "4%")
col4.metric("Total Ad Clicks", clicks) # "4%")

# Row B
col1, col2 = st.columns((6,4))
with col1:
    st.plotly_chart(fig3)
with col2:
    st.plotly_chart(fig4)
    
# Row C
# st.markdown('### Line chart')
col1, col2, col3 = st.columns((3,3,3))
with col1:
    st.pyplot(fig2)