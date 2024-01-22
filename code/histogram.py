import pandas as pd
import matplotlib.pyplot as plt

"""
csv_path = "/Users/skyler/Desktop/QuoteLLM/results3.0/CSVs/"
csv_file = csv_path + "works-from-OpenAI-lawsuit-results.csv"
graph_title = "John Grisham final-transcripts Results"
#graph_path = "/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/levenshtein_histograms/"
graph_path = "/Users/skyler/Desktop/"
graph_filename = graph_path + "john-grisham-3.5-histogram.png"

# make histogram
df = pd.read_csv(csv_file)
df = df.sort_values('start_token')
#df.to_csv(csv_file)

#filter dataframe by title if needed
#filtered_df = df[df['file'].apply(lambda x: 'The Firm' in x)].loc[:, 'levenshtein_distance']
filtered_df = df[df['file'].apply(lambda x: 'The Firm' in x)]
filtered_df.to_csv('/Users/skyler/Desktop/grisham_3.5.csv')
filtered_df = pd.read_csv('/Users/skyler/Desktop/grisham_3.5.csv')
#y = filtered_df['levenshtein_distance']
#y = df['levenshtein_distance']
plt.figure(figsize=(20, 6))
plt.hist(filtered_df['levenshtein_distance'])
# plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
#plt.hist(y)
plt.xlabel('Levenshtein Distance')
plt.ylabel('Number of Indices')
plt.title(graph_title)
plt.savefig(graph_filename)
plt.show()
"""
directory = "constitution"
csv_path = "/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/"  # sending csv to results path
csv_file = csv_path + f"{directory}-results.csv"
models_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

# separate into three dfs to make histograms
for model in models_list:
    df = pd.read_csv(csv_file)
    model_df = df[df['model'].eq(model)]
    print(model_df)

    # set up filtered and bad csv filepaths
    csv_file_filtered = csv_path + f"{directory}-{model}-results-filtered.csv"
    csv_file_sorry = csv_path + f"{directory}-{model}-results-bad.csv"

    # set up graph paths and title
    graph_path = f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/histograms/{model}/"  # sending graphs to results path (in graph directory)
    graph_filename = graph_path + f"{directory}-{model}-histogram.png"
    graph_filename_filtered = graph_path + f"{directory}-{model}-histogram-filtered.png"
    graph_title = f"{directory} with {model}"

    # make histogram with bad results included (unfiltered)
    y = model_df['levenshtein_distance']
    plt.figure(figsize=(20, 6))
    # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
    plt.hist(y)
    plt.xlabel('Levenshtein Distance')
    plt.ylabel('Number of Indices')
    plt.title(graph_title)
    plt.savefig(graph_filename)
    plt.show()

    # find the number of rows with bad answers, put these in a separate csv
    bad_rows = model_df[model_df['full_pred'].str.contains("I'm sorry", case=True)]
    print(bad_rows)
    bad_rows2 = model_df[model_df['full_pred'].str.contains("Sorry", case=True)]
    bad_rows3 = model_df[model_df['full_pred'].str.contains("I apologize", case=True)]
    print(bad_rows3)
    sorrydf = pd.concat([bad_rows3, bad_rows2, bad_rows])
    sorrydf.to_csv(csv_file_sorry)
    percent_bad = (len(bad_rows) + len(bad_rows2) + len(bad_rows3)) / len(model_df)

    # filter out all the bad results from the results file, make a new file, add the percent bad as a new column
    # filter out im sorry from model and save to im_sorry_df
    im_sorry_df = model_df[~model_df['full_pred'].str.contains("I'm sorry", case=True)]
    # filter out sorry from im_sorry
    sorry_df = im_sorry_df[~im_sorry_df['full_pred'].str.contains("Sorry", case=True)]
    # filter out i apologize
    filtered_df = sorry_df[~sorry_df['full_pred'].str.contains("I apologize", case=True)]
    filtered_df["Percent Bad"] = percent_bad
    filtered_df.to_csv(csv_file_filtered)

    # graph histogram from the filtered csv file
    y = filtered_df['levenshtein_distance']
    plt.figure(figsize=(20, 6))
    # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
    plt.hist(y)
    plt.xlabel('Levenshtein Distance')
    plt.ylabel('Number of Indices')
    plt.title(graph_title + " Filtered Responses")
    plt.savefig(graph_filename_filtered)
    plt.show()