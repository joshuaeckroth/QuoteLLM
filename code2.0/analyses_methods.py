# show how to filter bad rows
# also adding the additional metrics
# write a shell script


# histograms
# 2D histograms
# scatter plots
# ecdf plots
# sentence length plots
# new plot

# parameters: directory, model, metric/metrics

import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns


def create_histogram(df, model, category, metric, filepath):
    y = df[f'{metric}']
    plt.figure(figsize=(20, 6))
    plt.hist(y)
    plt.xlabel(f'{metric.title()}')
    plt.ylabel('Number of Indices')
    plt.title(f"{metric.title()} Distribution for {category.title()} with {model}")
    plt.savefig(filepath)


def create_2dhist(df, model, category, metric, filepath):
        file_data = []

        for

        mean = df[f'{metric}'].mean()
        print(mean)
        file_data.append((mean, title))

        # create a dataframe for exact metric distances and the file's mean ranking as compared to rest of files
        df_sorted = pd.DataFrame(columns=['ranked_mean', f'{metric}'])
        mean_ranking = 0

        # sort the means (ascending)
        file_data = sorted(file_data, key=lambda tup: tup[0])
        print(file_data)
        for pair in file_data:
            df = pd.read_csv(filepath + pair[1])
            distances = df[f'{metric}']
            print(distances)
            print(len(df[f'{metric}']))
            print(mean_ranking)
            print(pair[0])
            # loop through distance column
            for dist in distances:
                # new_row = [pair[0], x] # in row is the mean for whole file and independent distance
                # not appending mean itself to prevent means being binned together (each should be separate)
                new_row = [mean_ranking, dist]
                df_sorted.loc[len(df_sorted)] = new_row
            mean_ranking += 1

        print(mean_ranking)
        # Normalizing data:
        # each bin value is the percentage of items in the category that are in that bin
        # ex. in this bin, there are x items/ # items in the category (the file, also the row)
        # bin them all, then label each record as being in a bin (count items per bin)
        # then per file(row), get the percent per bin (# items in bin/ # items in the file(row))
        # plt.hist2d(df_sorted['levenshtein_distance'], df_sorted['ranked_mean'], bins=(10, mean_ranking), cmap = 'BuPu')
        # hist, xedges, yedges = np.histogram2d(df_sorted['ranked_mean'], df_sorted['levenshtein_distance'],  bins=[mean_ranking, 10])
        hist, xedges, yedges = np.histogram2d(df_sorted['ranked_mean'], df_sorted['rougeL'], bins=[mean_ranking, 10])
        # numbers bin divided by total number of reps for that category

        # Access, print, count items per bin
        for i in range(len(xedges) - 1):
            for j in range(len(yedges) - 1):
                count = hist[i, j]
                print(f"Bin ({i}, {j}): Count = {count}")

        print(df_sorted['ranked_mean'].value_counts())  # the sum of each row
        # Normalize the histogram by row (# of items in the file/category)

        print(hist)  # the sum of each row (matches ranked_mean value counts as above)
        print(hist.sum(axis=1, keepdims=True))  # this is correct
        hist_normalized = hist / hist.sum(axis=1, keepdims=True)
        # hist_normalized = hist / df_sorted['ranked_mean'].value_counts().tolist()
        # Convert hist_normalized to percentages
        for i in range(len(xedges) - 1):
            for j in range(len(yedges) - 1):
                percentage = hist_normalized[i, j] * 100  # Convert to percentage
                print(f"Bin ({i}, {j}): Percentage = {percentage:.2f}%")

        # Plotting data:
        plt.figure(figsize=(15, 6))
        plt.imshow(hist_normalized, origin='lower', cmap='BuPu', extent=[xedges[0], 10, yedges[0], mean_ranking])
        plt.colorbar(label='Frequency')
        # plt.title(f'Distribution of Levenshtein Distances per Category for {model}') # Density heatmap
        plt.title(f'Distribution of BLEU Distances per Category for {model}')  # Density heatmap
        # plt.xlabel('Levenshtein Distance')
        plt.xlabel('BLEU')
        plt.ylabel('Category (Ranked by Mean)')
        x_bins = np.arange(0, 10)  # 0-9
        y_bins = np.arange(0, mean_ranking)  # 0-8
        plt.xticks(x_bins.tolist(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        # make y-ticks show filename:
        # separate tuples into two lists (can't modify tuples, have to make new list)
        means, filenames = zip(*file_data)
        cut_names = []
        for name in filenames:
            # Split the filename and get first part
            cut_name = name.split('-gpt')[0]
            cut_names.append(cut_name)
        plt.yticks(y_bins.tolist(), cut_names)
        # save and show results
        plt.savefig(f'results4.0/visualization/categories-2d-histogram-{model}.png')
        plt.show()


def ecdf_plt(file, metric, graph_path, data_path):
    model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

    df_all = pd.DataFrame()
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    color_title = {}

    for model in model_list:

        graph_title = f"{metric} Comparison Scores with {model}"

        filenames = glob.glob(f'..all-models-results/CSVs/{model}/*.csv3')

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
            genre1 = file.split("-gpt")[0]  # Sci-Fi # no-author-fantasy
            # genre =  genre1.split(f"{prompt_type}-")[-1] # fantasy

            title = genre1.split("-")
            spaced_title = " ".join(title)
            caps_title = spaced_title.title()
            # graph_name = genre+"-density-plot.csv"
            # graph_title = "Cosine Vector Comparison Scores for " + caps_title

            optimal_scores = df[f"{metric}"]

            # df2 = pd.DataFrame(columns=['cosine'])
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

            ecdf = ECDF(scores)
            ecdf_sum = np.sum(ecdf(np.arange(0.0, 1.05, 0.001)))
            table_data.append([caps_title, ecdf_sum * 0.001])
            print()
            # table_data.append([caps_title, ecdf_sum])
            # legend_data.append([handle, ecdf_sum])
            legend_data.append([color_title[caps_title], ecdf_sum])

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
        sorted_legend = sorted(legend_data, key=lambda tup: tup[1], reverse=True)
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
        plt.savefig(filepath)
        plt.show()

        # format as latex table
        print(table_data)

        # TODO: make new method and call to generate datafile
        df = pd.DataFrame(table_data, columns=['Category', f'{model} scores'])
        print(df)
        if model == "gpt-3.5-turbo":
            df_all['Category'] = df['Category']
            # df_all = df_all.merge(df, on = 'Category', how = 'left')
            df_all = df
        else:
            pd.concat([df_all, df])
        # with open(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/ecdfs/cosine_ecdfs/cosine-ecdf-table-{model}.tex', 'w') as tf:
        with open(
                f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/ecdfs/{metric}-ecdf-table-{model}.tex',
                'w') as tf:
            tf.write(df.to_latex(index=False))



if __name__ == '__main__':

    file = '../all_data.csv'
    df = pd.read_csv(file) # put under CSVs

    models = df['model'].unique()
    categories = df['category'].unique()
    metric = ['levenshtein', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'optimal_cosine']

    for model in models:
        df = df[df['model'] == model]
        ecdf_plt(file, metric, f'../{model}/{metric}/ecdf.png', f'../{metric}_ecdf_data.csv')
        for category in categories:
            df = df[df['category'] == category]
            for metric in metric:
                create_histogram(df, model, category, metric, filepath)
                create_2dhist(df, model, category, metric, filepath)


    for metric in metric: # TODO: make all ecdf scores across the 3 models go to the ecdf csv in the ecdf method
            line_distribution(f'../{metric}_ecdf_data.csv', metric, f'../{metric}_line_distribution.png') #
filepath = '../{model}/{metric}/ecdf_data.csv'
pd.read_csv(f'{filepath}')



