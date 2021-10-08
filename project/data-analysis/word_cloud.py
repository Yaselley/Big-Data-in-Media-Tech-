# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:01:36 2021

@author: Yasel
"""
#%%

## Data Distribution 
train_df = pd.read_csv("/content/drive/MyDrive/alldatatest.csv")
test_df = pd.read_csv("/content/drive/MyDrive/alldatatest.csv")

import matplotlib.patches as mpatches
fig, ax= plt.subplots(figsize =(5,5))

ax = sns.countplot(x='label', data=train_df, palette=['#DC143C',"#32CD32"]);
for p in ax.patches:
    ax.annotate(p.get_height(), (p.get_x(), p.get_height()))

patch1 = mpatches.Patch(color='#DC143C', label='Negative')
patch2 = mpatches.Patch(color="#32CD32", label='Positive')

plt.legend(handles=[patch1, patch2], 
           bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("Distribution of the labels")
plt.show()
#%%
#%%
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator

original_column = 'review_punct' ##### COLUMN TO BE PREPROCESSED ("text" or "selected_text") !!!! ####
train_dfS = train_df[:10000]
# Start with one review:
df_positive = train_dfS[train_dfS['label']==1]
df_negative = train_dfS[train_dfS['label']==0]

tweet_all = " ".join(review for review in train_dfS[original_column])
tweet_positive = " ".join(review for review in df_positive[original_column])
tweet_negative = " ".join(review for review in df_negative[original_column])

fig, ax = plt.subplots(3, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_aLL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_all)
wordcloud_positive = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_positive)
wordcloud_negative = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_negative)

# Display the generated image:
ax[0].imshow(wordcloud_aLL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30, pad=25)
ax[0].axis('off')
ax[1].imshow(wordcloud_positive, interpolation='bilinear')
ax[1].set_title('Tweets under positive Class',fontsize=30, pad=25)
ax[1].axis('off')
ax[2].imshow(wordcloud_negative, interpolation='bilinear')
ax[2].set_title('Tweets under negative Class',fontsize=30, pad=25)
ax[2].axis('off')
plt.show()

