import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

for model in model_list:
    filepath = f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/'
    file_data = []

    # get mean levenshtein distance and filename from each file
    # add to list as tuple
    for file in glob.glob(filepath + '*'):
        df = pd.read_csv(file)
        # get mean distance for file
        mean = df['levenshtein_distance'].mean()
        print(mean)
        # get filename from filepath
        filename = file.split('/')[-1]
        file_data.append((mean, filename))

    # Manipulating data for graphing:
    # create a dataframe for exact levenshtein distances and the file's mean ranking as compared to rest of files
    df_sorted = pd.DataFrame(columns=['ranked_mean', 'levenshtein_distance'])
    mean_ranking = 0

    # sort the means (ascending)
    file_data = sorted(file_data, key=lambda tup: tup[0])
    print(file_data)
    # loop through files now sorted by mean
    for pair in file_data:
        # reopen the file from filename
        df = pd.read_csv(filepath + pair[1])
        distances = df['levenshtein_distance']
        print(distances)
        # distances = distances /len(df['levenshtein_distance'])
        print(len(df['levenshtein_distance']))
        print(mean_ranking)
        print(pair[0])
        # loop through distance column
        for dist in distances:
            # new_row = [pair[0], x] # in row is the mean for whole file and independent distance
            # append each levenshtein distance and file mean ranking (not the mean itself) as a row
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
    hist, xedges, yedges = np.histogram2d(df_sorted['ranked_mean'], df_sorted['levenshtein_distance'],  bins=[mean_ranking, 10])
    # numbers bin divided by total number of reps for that category

    # Access, print, count items per bin
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            count = hist[i, j]
            print(f"Bin ({i}, {j}): Count = {count}")

    print(df_sorted['ranked_mean'].value_counts()) # the sum of each row
    # Normalize the histogram by row (# of items in the file/category)

    print(hist) # the sum of each row (matches ranked_mean value counts as above)
    print(hist.sum(axis=1, keepdims=True)) # this is correct
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
    plt.title(f'Distribution of Levenshtein Distances per Category for {model}') # Density heatmap
    plt.xlabel('Levenshtein Distance')
    plt.ylabel('Category (Ranked by Mean Levenshtein Distance)')
    x_bins = np.arange(0, 10) # 0-9
    y_bins = np.arange(0, mean_ranking) # 0-8
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
    plt.savefig(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/2d_histograms/categories-2d-histogram-{model}.png')
    plt.show()


