import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class Cat_vs_cat_explorer():
    """
    class that holds functions to explore relations between Categorical Features and Categorical Target 
    """
    
    def __init__(self,df):
        """
        task: init the object
        parameters: df(pandas.DataFrame)
        return value:
        """
        
        self.df=df
        
    def explore(self, feature_name,target_name ):
        """
        task: create and show plots that visualize the 
        parameters: feature_name(String(name of the feature)), target_name(String(name of the target column))
        return value: 
        """
        
        #create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 6))
        order = sorted(self.df[feature_name].unique())

        #create a countplot on ax1 
        sns.countplot(feature_name, data = self.df, hue = target_name, ax = ax1, order = order).set_title("Counts For Feature:\n" + feature_name)

        #create a temporary df by first grouping after the feature variable and then calculate the percentages within each of those groups with respect to the target variable 
        df_temp = self.df.groupby(feature_name)[target_name].value_counts(normalize = True).\
        rename("percentage").\
        reset_index()
        df_temp["percentage"]=df_temp["percentage"]*100

        #create a barplot that uses the temporary dataframe as data and the target variable to color the bars
        fig = sns.barplot(x = feature_name, y = "percentage", hue = target_name, data = df_temp, ax = ax2, order = order)
        fig.set_ylim(0,100)

        #show the actual percentage value of each bar
        fontsize = 14 if len(order) <= 10 else 10
        for p in fig.patches:
            txt = str(p.get_height().round(2)) + '%'
            txt_x = p.get_x() 
            txt_y = p.get_height()
            fig.text(txt_x + 0.125, txt_y + 0.02,txt, fontsize = fontsize)

        #append a title to the plot
        ax2.set_title("Percentages For Feature: \n" + feature_name)




class Num_vs_cat_explorer():
    """
    holds functions for the exploration of relations between numerical features and categorical targets  
    """
    
    def __init__(self,df):
        """
        task: inits the object 
        parameters: df(pandas.DataFrame)
        return values:
        """
        
        self.df=df
        
    def explore(self, feature_name,target_name):
        """
        task: creates and shows a histogram, a KDE Plot, Boxplot and Violinplot
        parameters: feature_name(String(name of feature)),target_name(String(name of target variable))
        return value:
        """
        
        #create subplots 
        fig, axes = plt.subplots(1, 4, figsize = (25, 5))
        order = sorted(self.df[target_name].unique())

        #create the different plots
        sns.histplot(x = feature_name, hue = target_name, data = self.df, ax = axes[0])
        sns.kdeplot(x = feature_name, hue = target_name, data = self.df, fill = True, ax = axes[1])
        sns.boxplot(y = feature_name, hue = target_name, data = self.df, x = [""] * len(self.df), ax = axes[2])
        sns.violinplot(y = feature_name, hue = target_name, data = self.df, x = [""] * len(self.df), ax = axes[3])

        #set the titles of each graph
        fig.suptitle("For Feature:  " + feature_name)
        axes[0].set_title("Histogram For Feature " + feature_name)
        axes[1].set_title("KDE Plot For Feature " + feature_name)   
        axes[2].set_title("Boxplot For Feature " + feature_name)   
        axes[3].set_title("Violinplot For Feature " + feature_name)



class Cat_vs_num_explorer():
    """
    class that holds functions to explore relations bewteen a categorical Feature and Numerical Target
    """
    
    def __init__(self,df):
        """
        task: inits the object
        parameters: df(pandas.DataFrame)
        return value:
        """
        
        self.df=df
        
    def explore(self,feature_name,target_name):
        """
        task: creates and shows a countplot and boxplots
        parameters: feature_name(String(name of the feature)),target_name(String(name of the target))
        return value:
        """
        
        #create subplots 
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex = True)
        order = sorted(self.df[feature_name].unique())

        #create a countplot and boxplot
        sns.countplot(data = self.df, x = feature_name, ax = axes[0], order = order)   
        sns.boxplot(data = self.df, x = feature_name, ax = axes[1], y = target_name, order = order)

        #set the titles
        fig.suptitle("For Feature:  " + feature_name)
        axes[0].set_title("Countplot For " + feature_name)
        axes[1].set_title(feature_name + " --- " + target_name)

        #rotate the xticks
        for ax in fig.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)



class Num_vs_num_explorer():
    """
    class that holds basic functionality to explore realtionship between numerical feature and numerical target
    """
    
    def __init__(self,df):
        """
        task: inits the object
        parameters: df(pandas.DataFrame)
        return value:
        """
        
        self.df=df
        
    def explore(self,feature_name,target_name):
        """
        task: plots the correlation between feature and target
        parameters: feature_name(String(name of the feature)),target_name(String(name of the target))
        return value:
        """
        
        #calculate the correlation and set c to a corresponding color
        corr = self.df[[feature_name, target_name]].corr()[feature_name][1]    
        c = ["red"] if corr >= 0.7 else (["brown"] if corr >= 0.3 else\
                                        (["lightcoral"] if corr >= 0 else\
                                        (["blue"] if corr <= -0.7 else\
                                        (["royalblue"] if corr <= -0.3 else ["lightskyblue"]))))    

        #create subplots
        fig, ax = plt.subplots(figsize = (6, 6))

        #show scatterplot and set title
        sns.scatterplot(x = feature_name, y = target_name, data = self.df, c = c, ax = ax)        
        ax.set_title("Correlation between " + feature_name + " and " + target_name + " is: " + str(corr.round(4)))




class General_explorer():
    """
    class that holds basic functionality for general data exploration
    """
    
    def __init__(self,df):
        """
        task: inits the objects
        parameters: df(pandas.DataFrame)
        return value:
        """
        
        self.df=df
        
    def feature_distribution(self,feature_name):
        """
        task: creates and shows kdeplot,boxplot and probplot for the feature 
        parameters: feature_name(String(name of the feature))
        return value:
        """
        
        skewness = np.round(self.df[feature_name].skew(), 3)
        kurtosis = np.round(self.df[feature_name].kurtosis(), 3)


        fig, axes = plt.subplots(1, 3, figsize = (18, 6))

        sns.kdeplot(data = self.df, x = feature_name, fill = True, ax = axes[0], color = "orangered")
        sns.boxplot(data = self.df, y = feature_name, ax = axes[1], color = "orangered")
        stats.probplot(self.df[feature_name], plot = axes[2])

        axes[0].set_title("Distribution \nSkewness: " + str(skewness) + "\nKurtosis: " + str(kurtosis))
        axes[1].set_title("Boxplot")
        axes[2].set_title("Probability Plot")
        fig.suptitle("For Feature:  " + feature_name)
        
    def correlation_heatmap(self):
        """
        task: creates a correlation heatmaps for all numeric columns within the dataframe in self.df
        parameters:
        return value:
        """
        
        fig, ax = plt.subplots(figsize = (20, 20))
    
        sns.heatmap(self.df.corr(), cmap = "coolwarm", annot = True, fmt = ".2f", annot_kws = {"fontsize": 9},
                vmin = -1, vmax = 1, square = True, linewidths = 0.8, cbar = False)
    
    def get_num_vs_cat_columns(self):
        """
        task: return the names of numerical and categorical columns
        parameters: 
        return value: list[String(categorcial column)], list[String(numerical column)]
        """
        
        num_column_names= list(self.df.select_dtypes(include=[np.number]).columns.values)
        cat_column_names= [col for col in self.df.columns if col not in num_column_names]
        
        return cat_column_names,num_column_names
    
    def nan_correlation_heatmap(self):
        """
        task: create a Dataframe and Heatmap for the percentages of nan correlation between different columns 
        parameters:  
        return value: pd.DataFrame(Dataframe that holds the nan correlation info)
        """

        #create a dictionary that holds each column as a row and column
        nan_dict={"Column":[]}
        nan_cols=[col for col in self.df.columns if self.df[col].isna().any()]
        for nan_column in nan_cols:
            nan_dict[nan_column]=[]

        #iterate over the columns of self.df that contain nan values
        for curr_nan_column in nan_cols:
            #create a nan mask and sum up the nan values
            curr_nan=self.df[curr_nan_column].isna()
            curr_nan_sum= curr_nan.sum()
            nan_dict["Column"].append(curr_nan_column)
            
            #iterate over the columns of self.df that contain nan values
            for other_nan_column in nan_cols:
                #create a nan mask and sum up the nan values
                other_nan=self.df[other_nan_column].isna()
                other_nan_sum=other_nan.sum()
                
                #create a mask containing True if both columns are nan and sum the nan values 
                both_nan=np.logical_and(curr_nan,other_nan)
                both_nan_sum=both_nan.sum()
                
                #calculate the percentages and append those to the dict
                both_nan_percentage= round(both_nan_sum/curr_nan_sum *100,2)
                nan_dict[other_nan_column].append(both_nan_percentage)

        #create a DataFrame on basis of that dictionary
        nan_df= pd.DataFrame(nan_dict, index=nan_dict["Column"])
        nan_df.drop(columns="Column",inplace=True)

        #create and show a heatmap
        fig= plt.figure(figsize=(16,9))
        sns.heatmap(data=nan_df,annot=True,cmap='coolwarm',vmin=0,vmax=100,linewidth=4,linecolor='k')
        plt.title("NAN Correlation in percent")

        return nan_df

    def nan_info(self):
        """
        task: creates a Dataframe that contains nan count and nan percentage for each column in the self.df
        parameters: 
        return value: pd.DataFrame(DataFrame with the nan info) 
        """

        #create a dictionary and fill it with info for each column
        nan_dict={"Column":[],"NANs":[],"NAN_percentage":[]}
        for column_name in self.df.columns:
            nan_dict["Column"].append(column_name)

            #sum the nans and calc nan percentage, append both to the dict
            c=self.df[column_name].isna().sum()
            nan_dict["NANs"].append(c)
            nan_dict["NAN_percentage"].append(round(c/len(self.df[column_name].values)*100,2))

        #create a DataFrame on basis of that dict
        return pd.DataFrame(nan_dict)
     