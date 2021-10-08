# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:04:50 2021

@author: Yasel
"""

#%%
# Review Length 
## Review Length :
def plot_dist3(df, feature, title):
    '''
    Input:
        df: [Pandas] Dataset
        feature: [String] Column of tweets
    '''
    df['Character_Count'] = df[feature].apply(lambda x: len(str(x)))
    feature = 'Character_Count'
    # Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(18, 8))
    # Creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[:2, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 color='#e74c3c')
    ax1.set(ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))

    # Customizing the ecdf_plot.
    ax2 = fig.add_subplot(grid[2:, :2])
    # Set the title.
    ax2.set_title('Empirical CDF')
    # Plotting the ecdf_Plot.
    sns.distplot(df.loc[:, feature],
                 ax=ax2,
                 kde_kws={'cumulative': True},
                 hist_kws={'cumulative': True},
                 color='#e74c3c')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2.set(ylabel='Cumulative Probability')

    # Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    ax3.set_title('Box Plot')
    # Plotting the box plot.
    sns.boxplot(y=feature, data=df, ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=20))

    plt.suptitle(f'{title}', fontsize=24)
    
plot_dist3(train_dfS, "review_punct",'Characters per all Reviews for the column "text"')
#%%
def plot_word_number_histogram(textpo, textng,column_name):
    
    """A function for comparing word counts"""

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)
    sns.distplot(textpo.str.split().map(lambda x: len(x)), ax=axes[0], color='#e74c3c')
    sns.distplot(textng.str.split().map(lambda x: len(x)), ax=axes[1], color='#e74c3c')

    axes[0].set_xlabel('Word Count')
    axes[0].set_title('positive')
    axes[1].set_xlabel('Word Count')
    axes[1].set_title('negative')
    
    fig.suptitle('Word counts in tweets for the column "{}"'.format(column_name), fontsize=24, va='baseline')
    
    fig.tight_layout()
    
final_column = 'review_punct'
analysed_column = 'label'
    
plot_word_number_histogram(train_dfS[train_dfS[analysed_column] == 1][original_column],
                           train_dfS[train_dfS[analysed_column] == 0][original_column],original_column)

plot_word_number_histogram(train_dfS[train_dfS[analysed_column] == 1][final_column],
                           train_dfS[train_dfS[analysed_column] == 0][final_column],final_column)

#%%
from collections import Counter
import plotly.express as px

def top_most_common_words(dataset,column_name,top_nb,sentiment):
    """
    Inputs:
        Dataset: [Pandas]
        Column_name: [String] The name of the column we are interesting with
        sentiment: [String] "all", "positive", "neutral" or "negative"
    """
    dataset['temp_list'] = dataset[column_name].apply(lambda x:str(x).split())
    top = Counter([item for sublist in dataset['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(top_nb))
    temp.columns = ['Common_words','count']
    fig = px.bar(temp, x="count", y="Common_words",title='Common Words in {} for {} tweets'.format(column_name,sentiment), orientation='h', 
             width=700, height=700,color='Common_words',text='count')
    return fig.show()
    
top_nb=25
final_column = "review_stemmed"
top_most_common_words(train_dfS,final_column,top_nb,"all")
#%%