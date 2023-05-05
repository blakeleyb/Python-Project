# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 18:47:57 2023

@author: blake
"""

#Blakeley Baker
#Python Final Project -- Socioeconomic Factors and Opioid Overdose Deaths 
#in Virginia Counties (2016, 2018, 2020)

#import the necessary packages for data analysis
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
from datetime import datetime
import statsmodels.formula.api as smf
import plotly.io as pio
pio.renderers.default='browser' #makes plotly graphs pop up in browser


#Get working directory
os.getcwd()

#Change working directory to dataset location
os.chdir('C:/Users/blake/OneDrive/Desktop/Python/Project')

#Check that it was properly changed
os.getcwd()

#load csv data into environment as a pandas dataframe
opioid_data = pd.read_csv('Baker_Project_Proposal_Data.csv')


#Filter data to only include rows with an opioid overdose death count greater than 5
opioid_data = opioid_data[(opioid_data.DeathCount > 5)]

#Filter to get only the rows where Year = 2016 and reset index
opioid_2016 = opioid_data[(opioid_data.Year == 2016)]
opioid_2016.reset_index(inplace=True)

#Filter to get only the rows where Year = 2018 and reset index
opioid_2018 = opioid_data[(opioid_data.Year == 2018)]
opioid_2018.reset_index(inplace=True)

#Filter to get only the rows where Year = 2020 and reset index
opioid_2020 = opioid_data[(opioid_data.Year == 2020)]
opioid_2020.reset_index(inplace=True)

#Display all columns 
pd.set_option('display.max_columns', None)

#Get summary statistics for the dataset and include all of the summary statistics
opioid_data.describe(include = 'all')





#Creating Histograms
#First, set the theme of sns to darkgrid (not necessary-simply for aesthetics)
sns.set_theme(style = "darkgrid")

#Use matplotlip supplot function to create a figure with two rows and three columns
#Adjust figure size as well
fig, axs = plt.subplots(2, 3, figsize=(14, 10))

#Use seaborn histplot to create a histogram for PctMinority, PctUninsured, PctDisabled
#PctUnemployed, PctPov150, and DeathRate. Set kde to true in order to add a 
#kernal density estimate line. Change the color of the graph. Set ax equal to 
#axs followed by the coordinates where you want the graph to be located on the figure we 
#created in the last step. Finally, set the statistic to density. 
#creating seaborn histogram for six different variables, adding kde line, measuring density stat, adding to subplot
sns.histplot(data = opioid_data, x = "PctMinority", kde = True, color="blue", ax=axs[0,0], stat="density")
sns.histplot(data = opioid_data, x = "PctUninsured", kde = True, color="gold", ax=axs[0,1], stat="density")
sns.histplot(data = opioid_data, x = "PctDisabled", kde = True, color="green", ax=axs[0,2], stat="density")
sns.histplot(data = opioid_data, x = "PctUnemployed", kde = True, color="red", ax=axs[1,0], stat="density")
sns.histplot(data = opioid_data, x = "PctPov150", kde = True, color="pink", ax=axs[1,1], stat="density")
sns.histplot(data = opioid_data, x = "DeathRate", kde = True, color="purple", ax=axs[1,2], stat="density")

#Add title to figure
fig.suptitle("Histograms - Combined Years (2016, 2018, 2020)", size = 20, y = 0.92)

#Change all of the individual x-axis titles for the graphs
axs[0,0].set(xlabel='Minority Percentage (PctMinority)')
axs[0,1].set(xlabel='Uninsured Percentage (PctUninsured)')
axs[0,2].set(xlabel='Disabled Percentage (PctDisabled)')
axs[1,0].set(xlabel='Unemployed Percentage (PctUnemployed)')
axs[1,1].set(xlabel="Poverty Percentage (PctPov150)")
axs[1,2].set(xlabel="Death Rate (DeathRate)")

#Show the plot
plt.show()





#Creating a bumplot to visualize the themes variable 
#Use nlargest to get the ten largest numbers in the theme column for 
#opioid_2016, opioid_2018, and opioid_2020. This will give us the ten counties
#with the highest social vulnerability ranking for 2016, 2018, and 2020. 
themesa = opioid_2016.nlargest(n=10, columns=['Themes'])
themesb = opioid_2018.nlargest(n=10, columns=['Themes'])
themesc = opioid_2020.nlargest(n=10, columns=['Themes'])


#Combine newly created dataframes into a list
dfs = [themesa, themesb, themesc]

#Concatenate list to create a new dataframe with 30 rows (10 for each year)
themes = pd.concat(dfs)

#Use datetime function to turn Year column into a datetime object
themes["Year"] = pd.to_datetime(themes["Year"], format = '%Y')

#Create the bumpplot using px.line with the x-axis being the year, the y-axis
#being the Themes column, and color-code by locality
fig5 = px.line(themes, x = 'Year', y = 'Themes',
              color = 'Locality',
              color_discrete_sequence=px.colors.qualitative.Dark24, #choose color palette
              markers=True, #add markers
              hover_name = 'Themes', #can hover over line to see Themes data for a specific point
              title= "Themes-Top 10 Counties with Highest Rank") #add a title
fig5.update_traces(marker=dict(size=11)) #increase marker size
fig5.update_yaxes(title='Rank', #change y-axis title
                 visible=True, showticklabels=True) #show ticks 
fig5.update_xaxes(title='', visible=True, showticklabels=True) #erase x-axis title, show ticks
fig5.update_layout(xaxis=dict(showgrid=False), 
                  yaxis=dict(showgrid=False) ) #get rid of grid background
fig5.show() #cshow figure
fig5.write_html('bump_plot_themes.html', auto_open=True)


#Bivariate and Mulivariate OLS regression analysis
#Create bivariate OLS model with smf.ols. Save to object. Regress Themes onto DeathRate
my_models = smf.ols('DeathRate ~ Themes', data=opioid_data)
results = my_models.fit() #fit results
results.summary() #use summary to get the results from the object


#Create multivariate OLS model with smf.ols. Save to object. Regress Theme 1, 
#Theme2, Theme3, Theme4 onto DeathRate
#repeat steps above to get results
my_model2 = smf.ols('DeathRate ~ Theme1 + Theme2 + Theme3 + Theme4', data=opioid_data)
results2 = my_model2.fit() 
results2.summary()

#Repeat steps from my_model2 to get results
my_model3 = smf.ols('DeathRate ~ PctDisabled + PctUninsured + PctUnemployed + PctPov150 + PctMinority', data=opioid_data)
results3 = my_model3.fit()
results3.summary()




#Create scatterplots for correlated data

#First, set the theme of sns to darkgrid (not necessary-simply for aesthetics)
sns.set_theme(style = "darkgrid")

#Use matplotlip supplot function to create a figure with two rows and two columns
#Adjust figure size as well
fig, axs = plt.subplots(2, 2, figsize=(14, 8))

#use regplot to get a scatterplot with linear regression predicted trendline. 
#Repeat for all four variable with p<0.5
#Add title
sns.regplot(data=opioid_data, x='DeathRate', y='PctDisabled', ax=axs[0,0], color="red")
sns.regplot(data=opioid_data, x='DeathRate', y= 'PctPov150', ax=axs[0,1])
sns.regplot(data=opioid_data, x='DeathRate', y='Themes', ax=axs[1,0], color="green")
sns.regplot(data=opioid_data, x='DeathRate', y='Theme1', ax= axs[1,1], color="orange")
fig.suptitle("Scatterplots of Death Rate vs. Variables with p<0.05", size=20, y=0.92)
fig.show()




#import libraries below and use urlopen to get the json data from the github repository
#save into object ("counties")
from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
 counties = json.load(response)

#Convert FIPS to string from out dataset
opioid_data['FIPS'] = opioid_data['FIPS'].astype(str)

#create choropleth using px.choropleth. The map data comes from counties. Put this into 
#geojson. Use facet_col and facet_col_wrap to get the 3 years onto the same graph. 
#Use FIPS to match the counties in our dataset to the counties dataset.
#Change color scale, fill county based on Death Rate, use scope to look at USA
#add a title
#use update_geos to zoom in on Virginia
#save as html link
rank_fig = px.choropleth(opioid_data, geojson=counties, facet_col='Year', locations='FIPS', 
                         color='DeathRate',color_continuous_scale="spectral", scope='usa', 
                         facet_col_wrap=2, labels={"DeathRate": "Death Rate"}, 
                         title="Virginia Counties with 6 or more Opioid Overdose Deaths (2016, 2018, 2020)", 
                         hover_name = "Locality")
rank_fig.update_geos(fitbounds="locations")
rank_fig.write_html('choropleth.html', auto_open=True)
 
 




