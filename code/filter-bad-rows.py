import pandas as pd
import matplotlib.pyplot as plt
import glob

# filter the already filtered csv files containing additional metrics again, still save bad results to new files
models_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

for model in models_list:
    csv_path = f"/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/"  # sending csv to results path
    for csv_file in glob.glob(csv_path + f"{model}/*.csv2"):
        # read additional metric file as a df
        df = pd.read_csv(csv_file)
        filename = csv_file.split('/')[-1]

        # set up file path for bad csv--same filename in bad folder
        csv_file_sorry = csv_path + f"bad-responses/{filename}" # different from others

        new_filename = filename.split('.csv')[0]
        csv_file_filtered = csv_path + f"{model}/{new_filename}.csv3"

        # find the number of rows with bad answers, put these in a separate csv
        bad_rows = df[df['full_pred'].str.startswith("I'm sorry")]
        print(bad_rows)
        bad_rows2 = df[df['full_pred'].str.startswith("Sorry")]
        bad_rows3 = df[df['full_pred'].str.startswith("I apologize")]
        bad_rows4 = df[df['full_pred'].str.startswith("I am sorry")]
        bad_rows4 = df[df['full_pred'].str.startswith("As an AI")]
        bad_rows5 = df[df['full_pred'].str.startswith("Unfortunately")]
        bad_rows6 = df[df['full_pred'].str.startswith("As a language model AI")]
        bad_rows7 = df[df['full_pred'].str.startswith("As an artificial intelligence")]

        print(bad_rows3)
        sorrydf = pd.concat([bad_rows7, bad_rows6, bad_rows5, bad_rows4, bad_rows3, bad_rows2, bad_rows])
        percent_bad = (len(bad_rows) + len(bad_rows2) + len(bad_rows3) + len(bad_rows4) + len(bad_rows5) + len(bad_rows6) + len(bad_rows7)) / len(df)

        #if percent_bad != 0:
        if 1:
            # if there are bad results, histogram will look different and do need to save bad and filtered results separately
            sorrydf.to_csv(csv_file_sorry)
            # filter out all the bad results from the results file, make a new file, add the percent bad as a new column
            # filter out im sorry from model and save to im_sorry_df
            im_sorry_df = df[~df['full_pred'].str.startswith("I'm sorry")]
            # filter out sorry from im_sorry
            sorry_df = im_sorry_df[~im_sorry_df['full_pred'].str.startswith("Sorry")]
            # filter out i apologize
            apologize_df = sorry_df[~sorry_df['full_pred'].str.startswith("I apologize")]
            filtered_df = apologize_df[~apologize_df['full_pred'].str.startswith("I am sorry")]
            filtered_df["Percent Bad"] = percent_bad
            filtered_df.to_csv(csv_file_filtered)