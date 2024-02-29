import ssl

ssl.OPENSSL_VERSION = ssl.OPENSSL_VERSION.replace("LibreSSL", "OpenSSL")
import openai
import os
import sys
import glob
import random
from Levenshtein import distance
import csv
import pandas as pd
import matplotlib.pyplot as plt
# import spacy
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import tiktoken
import re

# change prompt to random author and change cvs fle path to random author for next run

openai.api_key = os.environ["OPENAI_API_KEY"]
embedding_model = SentenceTransformer('intfloat/e5-small-v2')
#model="final-transcripts"
#model= "gpt-4-1106-preview"
# Load the English language model
# nlp = spacy.load("en_core_web_sm")
token_enc = tiktoken.get_encoding("cl100k_base")

# change csv variable, graph title and graph filename
# change transcript file path to specific works directory
# change repetitions if needed

# models_list = ["gpt-3.5-turbo"]

model = "gpt-3.5-turbo"

directories=["copyright-lawsuit-works-gpt-3.5-turbo-results-filtered.csv","published-post-model-gpt-3.5-turbo-results.csv", "standard-lorem-ipsum-passage-gpt-3.5-turbo-results.csv", "random-text-gpt-3.5-turbo-results-filtered.csv"]
# directories = ["constitution"]
authors = ['Sarah Silverman', 'Shakespeare', 'J.R.R Tolkien', 'Elin Hilderbrand', 'J.K. Rowling', 'Emily Dickinson']
# for getting text transcripts (two models have same transcripts)
for directory in directories:
    # set up filepaths and titles
    # transcript_path = f"/Users/skyler/Desktop/QuoteLLM/transcripts/{transcript_folder}/*" # for getting transcripts for each model
    #transcript_path = f"/Users/skyler/Desktop/QuoteLLM/transcripts/final-transcripts/" # get the transcript path
    df_fromfile = pd.read_csv(f"/QuoteLLM/all-models-results/CSVs/gpt-3.5-turbo/{directory}")
    csv_path = "/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/" # sending csv to results path
    csv_file = csv_path + f"no-author-{directory}-results.csv"
    #csv_file = csv_path + f"random-author-{directory}-results.csv"

    # put all models in same csv -- can filter by model later to make histograms
    with open(csv_file, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["model", "file", "randtoken", "randtoken_count", "gt", "gt_portion", "pred", "answer", "levenshtein_distance", "full_pred", "full_answer", "optimal_cosine", "optimal_index", "cosine_scores",
            "start_token", "end_token"])

        # get all the text files in the directory specified
        # specify directories in the previous list becasue prompt must change slightly from category to category
        #for row in df_fromfile.itterows():
        j=0
        while j < len(df_fromfile):
            gt_quote = token_enc.encode(df_fromfile.loc[j, 'gt'])
            begin_quote_tokens = df_fromfile.loc[j, 'gt_portion']
            begin_quote = token_enc.encode(df_fromfile.loc[j, 'gt_portion'])
            randtoken = df_fromfile.loc[j, 'randtoken']
            randtoken_count = df_fromfile.loc[j, 'randtoken_count']
            start_token = df_fromfile.loc[j, "start_token"]
            end_token = df_fromfile.loc[j, "end_token"]
            title = df_fromfile.loc[j, 'file']

            # for model in models_list:
            prompt = "Finish the rest of the paragraph."
            #index = random.randint(0, len(authors))
            #author = author[index]
            # prompt = f"Finish the rest of the paragraph, it is by {author}."
            print(model)
            try:
                print(prompt)
                print(model)
                print('Begin quote:', begin_quote_tokens)
                print()

                # put the model loop here
                messages = [
                {"role": "system",
                "content": prompt},
                {"role": "user", "content": token_enc.decode(begin_quote)}
            ]


                completions = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=50,
                    request_timeout= 60,
                    n=1,
                    stop=None,
                    temperature=1.0)


                pred = completions['choices'][0]['message']['content']
                print(pred)
                # get GPT prediction into tokenized form
                # pred_doc = nlp(pred)
                pred_doc = token_enc.encode(pred)
                # pred_tokens = [token.text for token in pred_doc]
                pred_tokens = [token_enc.decode([token]) for token in pred_doc]
                print('pred_token:', pred_tokens)


                trimmed_gt = token_enc.encode(df_fromfile.loc[j, 'answer'])
                answer = token_enc.decode(trimmed_gt)
                #trimmed_gt = gt_quote[gt_portion:] #end quote (answer)
                # trimmed_tokens = [token.text for token in trimmed_gt]
                trimmed_tokens = [token_enc.decode([token]) for token in trimmed_gt]

                # cut pred_tokens length to be comparable to trimmed_gt
                # if pred_tokens length > trimmed_tokens length, cut it to length of trimmed, and all other positions (gt_quote, end token) stay the same
                if (len(pred_tokens) > len(trimmed_tokens)):
                    pred_tokens = pred_tokens[:len(trimmed_tokens)]
                    print('pred_tokens cut length:', pred_tokens)
                    print('trimmed_tokens:', trimmed_tokens)
                # if opposite, cut trimmed length, and update end_token
                # don't cut gt_quote length, want to see what was originally supposed to happen and compare to what actually happened (with the pred/trimmed lengths)
                if (len(pred_tokens) < len(trimmed_tokens)):
                    trimmed_tokens = trimmed_tokens[:len(pred_tokens)]
                    print('trimmed tokens cut length:',trimmed_tokens)
                    print('pred_tokens:', pred_tokens)

                end_token = start_token + len(begin_quote) + len(pred_tokens)-1

                # calculate Levenshtein Distance
                dist = distance(pred_tokens, trimmed_tokens) / len(pred_tokens)
                print('Dist:',dist)
                print()

                # calculate cosine distance from embeddings (the full length of each quote)
                input_lengths = []
                pred = pred.split(" ")
                i = 2

                for i in range(len(pred) + 1):
                    sub_list = pred[:i]
                    substr = " ".join(sub_list)
                    input_lengths.append(substr)

                # Compute embedding for both lists
                scores = []
                for k in range(len(input_lengths)):
                    embedding_1 = embedding_model.encode(token_enc.decode(trimmed_gt), convert_to_tensor = True)
                    embedding_2 = embedding_model.encode(input_lengths[k], convert_to_tensor = True)
                    score = util.pytorch_cos_sim(embedding_1, embedding_2)
                    scores.append(score)

                # get optimal score and its index
                abs_scores = [abs(ele) for ele in scores]
                optimal_cosine = max(abs_scores)
                optimal_index = abs_scores.index(optimal_cosine) + 1

                csvwriter.writerow(
                    [model, title, randtoken, randtoken_count, token_enc.decode(gt_quote), begin_quote_tokens,
                    pred_tokens, trimmed_tokens, dist, token_enc.decode(pred_doc), token_enc.decode(trimmed_gt), optimal_cosine, optimal_index, scores, start_token, end_token])

            except Exception as e:
                # don't increment repetitions if exception happens, need to get to 200 readings
                if e:
                    print(e)
                    # repetitions -= 1 # re-do this repetition
                    # print('Repetition:', repetition)
                    print('Retrying after timeout error...')
                    time.sleep(180)
                else:
                    raise e
            j += 1
    # graphing histograms
    # df = df.sort_values('start_token')
    # df.to_csv(csv_file)

"""
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
        bad_rows= model_df[model_df['full_pred'].str.contains("I'm sorry", case=True)]
        print(bad_rows)
        bad_rows2= model_df[model_df['full_pred'].str.contains("Sorry", case=True)]
        bad_rows3 = model_df[model_df['full_pred'].str.contains("I apologize", case=True)]
        print(bad_rows3)
        sorrydf = pd.concat([bad_rows3, bad_rows2, bad_rows])
        percent_bad= (len(bad_rows)+len(bad_rows2)+len(bad_rows3))/len(model_df)

        if percent_bad != 0:
            # if there are bad results, histogram will look different and do need to save bad and filtered results separately
            sorrydf.to_csv(csv_file_sorry)
            # filter out all the bad results from the results file, make a new file, add the percent bad as a new column
            #filter out im sorry from model and save to im_sorry_df
            im_sorry_df=model_df[~model_df['full_pred'].str.contains("I'm sorry", case=True)]
            #filter out sorry from im_sorry
            sorry_df=im_sorry_df[~im_sorry_df['full_pred'].str.contains("Sorry", case=True)]
            #filter out i apologize
            filtered_df = sorry_df[~sorry_df['full_pred'].str.contains("I apologize", case=True)]
            filtered_df["Percent Bad"]= percent_bad
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

"""
