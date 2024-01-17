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

openai.api_key = os.environ["OPENAI_API_KEY"]
embedding_model = SentenceTransformer('intfloat/e5-small-v2')
#model="gpt-3.5-turbo"
#model= "gpt-4-1106-preview"
# Load the English language model
# nlp = spacy.load("en_core_web_sm")
token_enc = tiktoken.get_encoding("cl100k_base")



# change csv variable, graph title and graph filename
# change transcript file path to specific works directory
# change repetitions if needed

models_list= ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]

for model in models_list:

    print(model)

    directories=["fantasy","nytimes-bestselling-romance"]
    for directory in directories: 

        csv_path = f"/home/haleystinebrickner/QuoteLLM/results-2024-{model}/"
        csv_file = csv_path + f"{directory}-results.csv"
        csv_file_2= csv_path + f"{directory}-results-im-sorry.csv"
        csv_file_sorry= csv_path + "im-sorry-bad.csv"
        graph_title = f"{directory}"
        graph_path = f"/home/haleystinebrickner/QuoteLLM/results-2024-graphs-{model}/"
        graph_filename = graph_path + f"{directory}-Histogram.png"
        graph_filename_2= graph_path + f"{directory}-Histogram-im-sorry.png"
        
        with open(csv_file, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                ["model", "file", "randtoken", "randtoken_count", "gt", "gt_portion", "pred", "answer", "levenshtein_distance", "full_pred", "full_answer", "optimal_cosine", "optimal_index", "cosine_scores",
                "start_token", "end_token"])

        
        

        

            for transcript_file in glob.glob(f"transcripts/{directory}/*"):
                print(transcript_file)
                file_name= transcript_file.split("/")[-1]
                title_author= file_name.split(".txt")[0]
                
                title_author2 = re.sub(r'([a-z])([A-Z])', r'\1 \2', title_author)

                author1= title_author.split("-")[-1] 
                author = re.sub(r'([a-z])([A-Z])', r'\1 \2', author1)
                title1= title_author.split("-")[0]
                title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title1)
                print(title)
                print(author)
                
                prompt= f"Finish the rest of the paragraph. It is from {title} by {author}."
                print(prompt)

                #for bible versions comment out author and remove from prompt

                token_count = 0
                time.sleep(30)
                with open(transcript_file) as t:
                    [title, transcript] = t.read().split("\n\n", 1)
                    transcript_lines = transcript.split("\n")
                    # doc = nlp(transcript)
                    doc = token_enc.encode(transcript)
                    print(doc)
                    print(type(doc))

                    repetitions = 200
                    for repetition in range(repetitions):
                        try:
                            # Get a random token index
                            randtoken = random.randint(0, len(doc) - 21)
                            # token = doc[randtoken].text
                            starting_point= [doc[randtoken]]
                            token = token_enc.decode(starting_point)
                            print(token)
                            # Get a random number for the substring length
                            randtoken_count = random.randint(20, 40)

                            # Create a substring
                            start_token = randtoken-1
                            end_token = start_token + randtoken_count
                            gt_quote = doc[start_token:end_token]  # this is a string
                            if (len(gt_quote) < 10):
                                continue # skip this iteration because it gets funky
                            print('Gt quote:', gt_quote)

                            gt_portion = random.randint(5, int(0.5 * len(gt_quote)))
                            begin_quote = gt_quote[:gt_portion]  # this is a string
                            # begin_quote_tokens = [token.text for token in begin_quote]
                            begin_quote_tokens = [token_enc.decode([token]) for token in begin_quote]
                            print('Begin quote:', begin_quote_tokens)
                            print()

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

                            trimmed_gt = gt_quote[gt_portion:] #end quote (answer)
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

                            print('Repetition:', repetition)
                            # increment repetitions if try works
                            repetitions += 1

                        except Exception as e:
                            # don't increment repetitions if exception happens, need to get to 200 readings
                            if e:
                                print(e)
                                repetitions -= 1 # re-do this repetition
                                print('Repetition:', repetition)
                                print('Retrying after timeout error...')
                                time.sleep(180)
                            else:
                                raise e


        # make histogram with i'm sorry
        df = pd.read_csv(csv_file)

        df = df.sort_values('start_token')
        df.to_csv(csv_file)
        y = df['levenshtein_distance']
        plt.figure(figsize=(20, 6))
        # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
        plt.hist(y)
        plt.xlabel('Levenshtein Distance')
        plt.ylabel('Number of Indices')
        plt.title(graph_title)
        plt.savefig(graph_filename)
        plt.show()

        phrase= "I'm sorry"
        bad_rows= df[df['full_pred'].str.contains("I'm sorry", case=True)]
        print(bad_rows)
        bad_rows2= df[df['full_pred'].str.contains("Sorry", case=True)]
        print(bad_rows2)
        percent_bad= (len(bad_rows)+len(bad_rows2))/len(df)
        sorrydf= pd.concat[bad_rows2, bad_rows]
        sorrydf.to_csv(csv_file_sorry)

        filtered_df=df[~df['full_pred'].str.contains("I'm sorry", case=True)]
        df=df[~df['full_pred'].str.contains("Sorry", case=True)]
        df["Percent Bad"]= percent_bad
        df.to_csv(csv_file_2)
        #df = pd.read_csv(csv_file_2)

        y = df['levenshtein_distance']
        plt.figure(figsize=(20, 6))
        # plt.hist(y, bins = np.arange(min(y), max(y) + 25, 25))
        plt.hist(y)
        plt.xlabel('Levenshtein Distance')
        plt.ylabel('Number of Indices')
        plt.title(graph_title)
        plt.savefig(graph_filename_2)
        plt.show()




