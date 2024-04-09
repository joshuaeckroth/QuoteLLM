import pandas as pd
import matplotlib.pyplot as plt
import glob

model_list = ["gpt-3.5-turbo" , "gpt-4-1106-preview", "gpt-4"]

for model in model_list:

    filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/gpt-results.csv')
    for filename in filenames:
        df = pd.read_csv(filename)
        df = df[df['model'] == model]
        df.to_csv(filename)
        # make histogram with bad results included (unfiltered)
        y = df['levenshtein_distance']
        plt.figure(figsize=(20, 6))
        # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
        plt.hist(y)
        plt.xlabel('Levenshtein Distance')
        plt.ylabel('Number of Indices')
        plt.title(f"Levenshtein Distance Distribution for GPT Story with {model}")
        plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/histograms/gpt-story-{model}-histogram.png")
        plt.show()  # display
"""
model_list = ["gpt-3.5-turbo"]
prompt_type = "no-author"
for model in model_list:

    #filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/*.csv3')
    filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{prompt_type}-*')
    for filename in filenames:
        df = pd.read_csv(filename)
        file = filename.split("/")[-1]
        genre1 = file.split("-gpt")[0] #Sci-Fi # no-author-fantasy
        genre = genre1.split(f"{prompt_type}-")[-1] # fantasy

        title = genre.split("-")
        spaced_title = " ".join(title)
        caps_title = spaced_title.title()

        # make histogram with bad results included (unfiltered)
        y = df['levenshtein_distance']
        plt.figure(figsize=(20, 6))
        # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
        plt.hist(y)
        plt.xlabel('Levenshtein Distance')
        plt.ylabel('Number of Indices')
        plt.title(f"Levenshtein Distance Distribution for {caps_title} with {model}")
        plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/{prompt_type}-visualization/{genre}_histogram-{prompt_type}.png")
        plt.show()  # display
"""

"""
# filter CSVS with bad results (i am sorry, sure, etc)
directories = ["popular-slogan","bible-versions"]

for directory in directories:
    csv_path = "/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/"  # sending csv to results path
    csv_file = csv_path + f"{directory}-results.csv"
    models_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

    # separate into three dfs to make histograms
    for model in models_list:
        df = pd.read_csv(csv_file)
        model_df = df[df['model'].eq(model)]
        print(model_df)

        # set up filtered and bad csv filepaths
        csv_file_filtered = csv_path + f"{model}/{directory}-{model}-results-filtered.csv"
        csv_file_sorry = csv_path + f"bad-responses/{directory}-{model}-results-bad.csv"

        # set up graph paths and title
        graph_path = f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/histograms/"  # sending graphs to results path (in graph directory)
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
        bad_rows4 = model_df[model_df['full_pred'].str.contains("I am sorry", case=True)]
        print(bad_rows3)
        sorrydf = pd.concat([bad_rows4, bad_rows3, bad_rows2, bad_rows])
        percent_bad = (len(bad_rows) + len(bad_rows2) + len(bad_rows3) + len(bad_rows4)) / len(model_df)

        if percent_bad != 0:
            # if there are bad results, histogram will look different and do need to save bad and filtered results separately
            sorrydf.to_csv(csv_file_sorry)
            # filter out all the bad results from the results file, make a new file, add the percent bad as a new column
            # filter out im sorry from model and save to im_sorry_df
            im_sorry_df = model_df[~model_df['full_pred'].str.contains("I'm sorry", case=True)]
            # filter out sorry from im_sorry
            sorry_df = im_sorry_df[~im_sorry_df['full_pred'].str.contains("Sorry", case=True)]
            # filter out i apologize
            apologize_df = sorry_df[~sorry_df['full_pred'].str.contains("I apologize", case=True)]
            filtered_df = apologize_df[~apologize_df['full_pred'].str.contains("I am sorry", case=True)]
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
        else: # no bad results for this model, save the model df plainly
           # model_df.to_csv(csv_path + f"{model}/{directory}-{model}-results.csv")

"""