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
df_all = pd.DataFrame()
color_title = {}
metric = "bleu"

for model in model_list:
    # for all lines different colors
    graph_filename = f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/ecdfs/{metric}-ecdf-plot-{model}.png'
    # for some lines highlighted
    # graph_filename = '/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine-ecdf-plot-refined.png'
    # df = pd.read_csv('/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/bible-Versions-results.csv')

    graph_title = f"{metric} Scores for {model}"
    # graph_filename = '/Users/skyler/Desktop/QuoteLLM/results2.0/density_plots/cosine-density-plot.png'
    filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/*.csv3')
    # get most recent files (the ones with the new metrics and filtered responses)
    plt.figure(figsize=(20, 6))

    pos = 0
    palette_pos = 0
    palettes = [sns.color_palette("deep"), sns.color_palette("pastel"), sns.color_palette("husl", 3)]
    # palette = palettes[0] # for when having all lines different colors
    palette = sns.color_palette()  # for when highlighting a few lines in particular

    table_data = []
    legend_data = []
    # trapz_table_data = []
    # simpson_table_data = []
    for filename in filenames:
        df = pd.read_csv(filename)
        file = filename.split("/")[-1]
        genre = file.split("-gpt")[0]  # Sci-Fi

        title = genre.split("-")
        spaced_title = " ".join(title)
        caps_title = spaced_title.title()
        # graph_name = genre+"-density-plot.csv"
        # graph_title = "Cosine Vector Comparison Scores for " + caps_title

        #optimal_scores = df["optimal_cosine"]
        # df2 = pd.DataFrame(columns=['cosine'])
        scores = df[f"{metric}"]
        """ 
        scores = []
        # get optimal score as a float
        for cos_score in optimal_scores:
            score = cos_score.split("tensor([[")
            score_split = score[1].split("]])")
            score_split = score_split[:len(score_split) - 1]
            str_score = score_split[0]
            opt_score = float(str_score)
            # print("Opt", opt_score)
            # df2['cosine'] = df2.append(opt_score, ignore_index=True)
            # print(df2['cosine'])
            scores.append(opt_score)
        """

        # print(scores)
        # make sure no color palettes repeat
        if pos > 9:
            pos = 0
            palette = palettes[palette_pos + 1]
        # plot the line
        color = palette[pos]
        if model == model_list[0]:
            color_title[caps_title] = color
        print(color_title)
        sns.ecdfplot(scores, label=caps_title, color=color_title[caps_title])
        pos += 1

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
        ecdf_sum = np.sum(ecdf(np.arange(0.0, 1.05, 0.001)))
        table_data.append([caps_title, ecdf_sum * 0.001])
        print()
        # table_data.append([caps_title, ecdf_sum])
        # legend_data.append([handle, ecdf_sum])
        legend_data.append([color_title[caps_title], ecdf_sum])

        # plt.hist()
        # ax.ecdf(scores,  label = caps_title, color = palette[pos])
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
    print("Sorted table")
    minimum_value = sorted_table[0][1]
    maximum_value = sorted_table[-1][1]
    print(sorted_table)
    print(minimum_value)
    print(maximum_value)
    for x in range(len(sorted_table)):
        sorted_table[x][1] = 1 - ((sorted_table[x][1] - minimum_value) / (maximum_value - minimum_value))
    minimum_value = sorted_table[0][1]
    maximum_value = sorted_table[-1][1]
    print(minimum_value)
    print(maximum_value)
    print(sorted_table)
    # sort legend data
    sorted_legend = sorted(legend_data, key=lambda tup: tup[1], reverse = True)
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

    # format as latex table
    print(table_data)
    df = pd.DataFrame(table_data, columns=['Category', f'{model} scores'])
    print(df)
    if model == "gpt-3.5-turbo":
        df_all['Category'] = df['Category']
    df_all = df_all.merge(df, on='Category', how='left')
    with open(
            f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/ecdfs/{metric}-ecdf-table-{model}.tex',
            'w') as tf:
        tf.write(df.to_latex(index=False))
with open(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/ecdfs/{metric}-ecdf-table.tex',
          'w') as tf:
    tf.write(df_all.to_latex(index=False))

    # format the table plot
    """
    plt.figure(figsize=(8, 8))  # Adjust figure size if needed
    table = plt.table(cellText=sorted_table, loc='center', cellLoc='center', colLabels=['Category', 'Area Under Curve'], edges='closed')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Adjust the scale of the table if needed
    plt.axis('off')
    plt.title(f'Table of Areas under Empirical CDF Curves for {model}')
    #plt.savefig(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/cosine_ecdfs/cosine-ecdf-table-{model}.png')
    #plt.show()
    """

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

# divide by range

# subtract


# store the sum of the ecdf function, then update the values by sutracting the min, and then divide that result by the max-min
# so the new max will be the 1.0 case

# this is just for the table not the graph