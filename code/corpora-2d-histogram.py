import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

for model in model_list:
    filepath = f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/standard-lorem'
    df_means = pd.DataFrame(columns=['title', 'mean_levenshtein'])
    # all distances under a title need to have the same mean (to rank the means of all titles)
    # loop through all files, putting each title and its respective mean levenshtein distance in a dataframe
    for file in glob.glob(filepath+'*'):
        print(file)
        df = pd.read_csv(file)
        filename = file.split('/')[-1]
        category = filename.split('-gpt')[0]
        # make df of all titles and their means
        means = df.groupby('file', as_index=False)['levenshtein_distance'].mean()
        means['csv'] = file.split('/')[-1] # filename only
        means.columns = ['title', 'mean_levenshtein', 'csv']
        # add means for the file to df for all files
        df_means = pd.concat([df_means, means])


    print(df_means)
    # after all files looped through, sort and rank means
    df_ranked_means = df_means.sort_values('mean_levenshtein')
    # reset index and use index as mean ranking method
    df_ranked_means = df_ranked_means.reset_index(drop=True)
    df_ranked_means['ranked_mean'] = df_ranked_means.index
    print(df_ranked_means)

    # need to get the individual levenshtein distances for each title alongside their ranked means
    # loop through all files again
    df_distances = pd.DataFrame()
    for file in glob.glob(filepath + '*'):
        df = pd.read_csv(file)
        # get list of titles for the file
        titles = df['file'].unique().tolist()
        # loop through each title and create dataframe for exact levenshtein distances
        for title in titles:
            # get all rows with the title
            df_title = (df.loc[df['file'] == title])
            # make dataframe of distances and the title
            df_title = pd.DataFrame(df_title, columns = ['file', 'levenshtein_distance'])
            # add number of exact distances as a column (for normalization)
            # df_title['item_count'] = len(df_title)
            # rename columns
            df_title.columns = ['title', 'levenshtein_distance']
            # add the title/distance data to a new dataframe for all titles
            df_distances = pd.concat([df_distances, df_title])

    print(df_distances)
    print(df_ranked_means) # the number of titles
    # merge exact distances and titles with ranked mean distances and titles
    df = pd.merge(df_ranked_means, df_distances, on="title")
    print(df)

    # now you can get ranked_mean and levenshtein_distance columns for histogram
    # also get titles for y-axis and item_count for normalization

    # Normalizing data:
    # each bin value is the percentage of items from one title that are in that bin
    # ex. in this bin, there are x items/ # items with one title from a category (the file column, also the row)
    # bin them all, then label each record as being in a bin (count items per bin)
    # then per title (row), get the percent per bin (# items in bin/ # items with the title (row))
    # plt.hist2d(sorted['levenshtein_distance'], sorted['mean'], bins=(10, mean_bin_count), cmap = 'BuPu')
    hist, xedges, yedges = np.histogram2d(df['ranked_mean'], df['levenshtein_distance'], bins=(len(df_ranked_means), 10))
    # numbers bin divided by total number of reps for that title

    print(df['ranked_mean'].value_counts().tolist()) # of items under a title (all have the same ranked_mean value)

    # Access, print, count items per bin
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            count = hist[i, j]
            print(f"Bin ({i}, {j}): Count = {count}")

    # Normalize the histogram by row (# of items in the file/category)
    print(hist.sum(axis=1, keepdims=True)) # matches rank_mean value counts (checking that binning is correct)
    hist_normalized = hist / hist.sum(axis=1, keepdims=True)

    # Convert hist_normalized to percentages
    for i in range(len(xedges) - 1):
        for j in range(len(yedges) - 1):
            percentage = hist_normalized[i, j] * 100  # Convert to percentage
            # print(f"Bin ({i}, {j}): Percentage = {percentage:.2f}%")

    # Plotting data:
    plt.figure(figsize=(20, 8))
    plt.imshow(hist_normalized, origin='lower', cmap='BuPu', extent=[xedges[0], 10, yedges[0], len(df_ranked_means)])
    plt.colorbar(label='Frequency')
    plt.title(f'Distribution of Levenshtein Distances per Title Published Post-{model}') # Density heatmap
    plt.xlabel('Levenshtein Distance')
    plt.ylabel('Title (Ranked by Mean Levenshtein Distance)')
    x_bins = np.arange(0, 10) # 0-9
    y_bins = np.arange(0, len(df_ranked_means)) #
    plt.xticks(x_bins.tolist(), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # make y-ticks show title/filename:
    cut_names = []
    # multiple shakespeare files = multiple of same title
    # full titles not printed
    # some text in titles -- could change title manually or write code to delete after certain point
    # scale of graph
    for title in df_ranked_means['title']:
        # Split the title by the number of words
        pieces = title.split()
        if len(pieces) < 4:
            cut_name = title
        else:
            cut_name = ""
            for i in range(0, 4):
                cut_name+=" ".join(pieces[i:i+1]) + " "
            #cut_name = (" ".join(pieces[i:i+3]) for i in range(0, len(pieces), 4))
            print(cut_name)
        cut_names.append(cut_name)
    print(cut_names)
    plt.yticks(y_bins.tolist(), cut_names)
    # save and show results
    plt.savefig(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/2d_histograms/{category}-{model}-2d-histogram.png')
    plt.show()
    print(len(xedges))
    print(len(yedges))

    # show titles not filenames
    # proportions
    # why nan percentages? dividing by 0 bin 127 128 118 81 40

    # create dataframe of mean levenshtein distances for each title
    # means = df.groupby('file', as_index=False)['levenshtein_distance'].mean()
    # create dataframe of the number of rows under each title (for normalization)
    # item_counts = pd.DataFrame(df['file'].value_counts())
    # join the dataframes by the file column
    # new_rows = pd.merge(means, item_counts, on="file")
    # get filename from filepath, add filename to each row from the file
    # filename = file.split('/')[-1]
    # new_rows['csv'] = filename
    # df_data = pd.concat([df_data, new_rows]) # add the file data to the larger df

    """
        df_data = pd.concat([df_data, means])
        # loop through titles and create a dataframe for each title's data
        # get list of titles for the file
        titles = df['file'].unique().tolist()
        print(titles)
        for title in titles:
            title_df = (df.loc[df['file'] == title])
            # create dataframe for title, individual levenshtein distances, and csv filename
            title_df = pd.DataFrame(title_df, columns = ['file', 'levenshtein_distance'])
            # title_df['mean']=title_df['levenshtein_distance'].mean()
            # get filename from filepath, add filename to each row from the file
            filename = file.split('/')[-1]
            title_df['csv'] = filename
            # rename columns for clarity
            title_df.columns = ['title', 'levenshtein_distance', 'csv']
            print(title_df)
            # add the title data to the dataframe for whole file
            file_data = pd.concat([file_data, title_df])
    
        # after looping through all the titles
        # get mean for each title in the file
        means = df.groupby('file', as_index=False)['levenshtein_distance'].mean()
        # rename columns for clarity
        means.columns = ['title', 'mean_levenshtein']
        print(means)
        # sort and rank the means before merging them into whole
    
        # merge the means into the file_data as a column
        print(file_data)
        new_rows = pd.merge(means, file_data, on="title")
        print(new_rows['mean_levenshtein'])
    """