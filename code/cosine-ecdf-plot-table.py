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

model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

for model in model_list:
    # for all lines different colors
    graph_filename = f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/cosine_ecdfs/cosine-ecdf-plot-{model}.png'
    # for some lines highlighted
    # graph_filename = '/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine-ecdf-plot-refined.png'
    # df = pd.read_csv('/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/bible-Versions-results.csv')

    graph_title = f"Cosine Vector Comparison Scores for {model}"
    #graph_filename = '/Users/skyler/Desktop/QuoteLLM/results2.0/density_plots/cosine-density-plot.png'
    filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/*') # get most recent files
    plt.figure(figsize=(20, 6))

    pos = 0
    palette_pos = 0
    palettes = [sns.color_palette("deep"), sns.color_palette("pastel"), sns.color_palette("husl", 3)]
    #palette = palettes[0] # for when having all lines different colors
    palette = sns.color_palette() # for when highlighting a few lines in particular

    table_data = []
    legend_data = []
    # trapz_table_data = []
    # simpson_table_data = []
    for filename in filenames:
        df = pd.read_csv(filename)
        file = filename.split("/")[-1]
        genre = file.split("-gpt")[0] #Sci-Fi

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

        if pos > 9:
            pos = 0
            palette = palettes[palette_pos+1]
        sns.ecdfplot(scores, label = caps_title, color = palette[pos])
        color = palette[pos]
        pos+=1

        # plot the line with specific colors for a few lines only
        """
        standout_categories = ['Quotes', 'Slogans', 'Constitution', 'Bible Versions', 'Recipes', 'Song Lyrics', 'Published 2023', 'Suing Works']
        if (standout_categories.count(caps_title) != 0):
            pos += 1
            sns.ecdfplot(scores, label=caps_title, color=palette[pos])
            color = palette[pos]
        else:
            sns.ecdfplot(scores, label=None, color=palette[0])
            color = palette[0]
        """

        ecdf = ECDF(scores)
        ecdf_sum = np.sum(ecdf(np.arange(0.0, 1.0, 0.05)))
        table_data.append([caps_title, ecdf_sum])
        # legend_data.append([handle, ecdf_sum])
        legend_data.append([color, ecdf_sum])

        #plt.hist()
        #ax.ecdf(scores,  label = caps_title, color = palette[pos])
        # alternate area under curve calculation method
        """
        trapz_area = np.trapz(scores, dx = 0.01)
        print(trapz_area)
        simpson_area = simpson(scores, dx = 0.01)
        print(simpson_area)
        trapz_table_data.append([caps_title, trapz_area])
        simpson_table_data.append([caps_title, simpson_area])
        """

    # sort the table_data for area method and use in ecdf legend
    sorted_table = sorted(table_data, key=lambda tup: tup[1])
    print(sorted_table)
    # sort legend data
    sorted_legend = sorted(legend_data, key=lambda tup: tup[1])
    print(legend_data)
    print(list(zip(*sorted_legend))[0])

    # add legend to plot
    plt.legend(list(zip(*sorted_table))[0], prop={'size': 8}, title='Category')
    ax = plt.gca()
    leg = ax.get_legend()
    # loop through legend sorted by ecdf values, and assign their respective colors
    for i in range(len(sorted_legend)):
        leg.legendHandles[i].set_color(sorted_legend[i][0])

    # format the complete plot
    plt.xlabel('Optimal Score')
    plt.ylabel('Proportion')
    plt.title(graph_title)
    plt.savefig(graph_filename)
    plt.show()

    # format the table plot
    print(table_data)
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    table = plt.table(cellText=sorted_table, loc='center', cellLoc='center', colLabels=['Category', 'Area Under Curve'], edges='closed')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Adjust the scale of the table if needed
    plt.axis('off')
    plt.title(f'Table of Areas under Empirical CDF Curves for {model}')
    plt.savefig(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/cosine_ecdfs/cosine-ecdf-table-{model}.png')
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
