import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
# assign specific colors to 2 or 3 lines
# calculate area under each curve -- trapezoidal function?
# sort the table by area
# sort the legend


# for all lines different colors
graph_filename = '/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine-ecdf-plot.png'
# for some lines highlighted
# graph_filename = '/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine-ecdf-plot-refined.png'
# df = pd.read_csv('/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/bible-Versions-results.csv')

graph_title = "Cosine Vector Comparison Scores"
#graph_filename = '/Users/skyler/Desktop/QuoteLLM/results2.0/density_plots/cosine-density-plot.png'
filenames = glob.glob('/Users/skyler/Desktop/QuoteLLM/results3.0/CSVs/*') # get most recent files
plt.figure(figsize=(20, 6))

pos = 0
palette_pos = 0
palettes = [sns.color_palette("pastel"), sns.color_palette("deep"), sns.color_palette("husl", 3)]
palette = palettes[0] # for when having all lines different colors
# palette = sns.color_palette() # for when highlighting a few lines in particular

table_data = []
# trapz_table_data = []
# simpson_table_data = []
for filename in filenames:
    df = pd.read_csv(filename)
    file = filename.split("/")[-1]
    genre = file.split("-results.csv")[0] #Sci-Fi

    title = genre.split("-")
    spaced_title = " ".join(title)
    caps_title = spaced_title.title()
    # graph_name = genre+"-density-plot.csv"
    # graph_title = "Cosine Vector Comparison Scores for " + caps_title


    optimal_scores = df["optimal_cosine"]
    #df2 = pd.DataFrame(columns=['cosine'])
    scores = []

    # get optimal score as a float
    for cos_score in optimal_scores:
        score = cos_score.split("tensor([[")
        score_split = score[1].split("]])")
        score_split = score_split[:len(score_split)-1]
        str_score = score_split[0]
        opt_score = float(str_score)
        # print("Opt", opt_score)
        # df2['cosine'] = df2.append(opt_score, ignore_index=True)
        # print(df2['cosine'])
        scores.append(opt_score)

    # print(scores)
    # plot the line with each line being a different color
    # """
    if pos > 9:
        pos = 0
        palette = palettes[palette_pos+1]
    sns.ecdfplot(scores, label = caps_title, color = palette[pos])
    # """

    # plot the line with specific colors for a few lines only
    """
    if (caps_title == 'Quotes'):
        sns.ecdfplot(scores, label=caps_title, color=palette[1])
    elif (caps_title == 'Slogans'):
        sns.ecdfplot(scores, label=caps_title, color=palette[2])
    elif (caps_title == 'Constitution'):
        sns.ecdfplot(scores, label=caps_title, color=palette[3])
    elif (caps_title == 'Bible Versions'):
        sns.ecdfplot(scores, label=caps_title, color=palette[4])
    elif (caps_title == 'Recipes'):
        sns.ecdfplot(scores, label=caps_title, color=palette[5])
    elif (caps_title == 'Song Lyrics'):
        sns.ecdfplot(scores, label=caps_title, color=palette[6])
    else:
        sns.ecdfplot(scores, label=None, color=palette[0])
    """

    # alternate area under curve calculation method
    """
    trapz_area = np.trapz(scores, dx = 0.01)
    print(trapz_area)
    simpson_area = simpson(scores, dx = 0.01)
    print(simpson_area)
    trapz_table_data.append([caps_title, trapz_area])
    simpson_table_data.append([caps_title, simpson_area])
    """
    ecdf = ECDF(scores)
    ecdf_sum = np.sum(ecdf(np.arange(0.0, 1.0, 0.05)))
    table_data.append([caps_title, ecdf_sum])
    #plt.hist()
    #ax.ecdf(scores,  label = caps_title, color = palette[pos])
    pos+=1

# format the complete plot
plt.legend(prop={'size': 8}, title='Category')
plt.xlabel('Optimal Score')
plt.ylabel('Proportion')
plt.title(graph_title)
plt.savefig(graph_filename)
plt.show()

# sort the table_data for area method
sorted_table = sorted(table_data, key=lambda tup: tup[1])
print(sorted_table)

# format the table plot
print(table_data)
plt.figure(figsize=(8, 8))  # Adjust figure size if needed
table = plt.table(cellText=sorted_table, loc='center', cellLoc='center', colLabels=['Category', 'Area Under Curve'], edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Adjust the scale of the table if needed
plt.axis('off')
plt.title('Table of Areas under Empirical CDF Curves')
plt.savefig('/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine-ecdf-table.png')
plt.show()

"""
# sort the table_data for trapz area method
sorted_table = sorted(trapz_table_data, key=lambda tup: tup[1])
print(sorted_table)

# format the table plot
print(trapz_table_data)
plt.figure(figsize=(8, 8))  # Adjust figure size if needed
table = plt.table(cellText=sorted_table, loc='center', cellLoc='center', colLabels=['Category', 'Area Under Curve'], edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Adjust the scale of the table if needed
plt.axis('off')
plt.title('Table of Areas under Empirical CDF Curves')
plt.savefig('/Users/skyler/Desktop/cosine-ecdf-trapz-table.png')
plt.show()

# sort the table_data for simpson area method
sorted_table = sorted(simpson_table_data, key=lambda tup: tup[1])
print(sorted_table)

# format the table plot
print(simpson_table_data)
plt.figure(figsize=(8, 8))  # Adjust figure size if needed
table = plt.table(cellText=sorted_table, loc='center', cellLoc='center', colLabels=['Category', 'Area Under Curve'], edges='closed')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.5)  # Adjust the scale of the table if needed
plt.axis('off')
plt.title('Table of Areas under Empirical CDF Curves')
plt.savefig('/Users/skyler/Desktop/cosine-ecdf-simpson-table.png')
plt.show()

    
=

optimal_scores = df["optimal_cosine"]
#df2 = pd.DataFrame(columns=['cosine'])
scores = []

# get optimal score as a float
for cos_score in optimal_scores:
    score = cos_score.split("tensor([[")
    score_split = score[1].split("]])")
    score_split = score_split[:len(score_split)-1]
    str_score = score_split[0]
    opt_score = float(str_score)
    print("Opt", opt_score)
    # df2['cosine'] = df2.append(opt_score, ignore_index=True)
    # print(df2['cosine'])
    scores.append(opt_score)

print(scores)

# make density plot
# can graph histogram and density plot separate or together, but y-axis changes
plt.figure(figsize=(10, 6))
sns.distplot(a=scores, hist = True, bins = 10, hist_kws={"edgecolor": 'white'})
plt.xlabel('Optimal Score')
plt.ylabel('Density')
plt.title(graph_title)
plt.savefig(graph_filename)
plt.show()
"""
