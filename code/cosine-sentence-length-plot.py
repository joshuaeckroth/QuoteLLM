import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
#gpt 4 preview has to go first so one of its start tokens is chosen, since its the smallest dataframe of valid results

# choose a row
row_num = 5
# don't go by row num go by start token
#choose csv
start_token = 0
end_token = 0
model_list = ["gpt-4-1106-preview", "gpt-3.5-turbo", "gpt-4"]
plt.xlabel("Token Count")
plt.ylabel("Cosine Score")

for model in model_list:
    filename = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/famous-quote*')
    print(filename[0])
    df = pd.read_csv(filename[0])
    file = filename[0].split("/")[-1]
    genre = file.split("-gpt")[0]  # "quotes"

    title = genre.split("-")
    spaced_title = " ".join(title)
    caps_title = spaced_title.title() # "Quotes"
    #suing-works
        # row number: 8
    # change row number to graph different quote (based on start token)
    if model == model_list[0]:
        start_token = df.iloc[row_num]["start_token"]
        end_token = df.iloc[row_num]["end_token"]
        print("Start token:", start_token)
        print("End token:", end_token)

    #always print this to check
    print(start_token)
    print(end_token)
    row = df.loc[df['start_token'] == start_token]
    row = row.loc[row['end_token'] == end_token]
    print()
    print("Row:")
    print(row)
    if row.empty:
        pass
    else:
        tensor_scores = row.iloc[0]["cosine_scores"]
        print(tensor_scores)
        optimal_score = row.iloc[0]["optimal_cosine"]
        print(optimal_score)
        print("Answer:")
        print(row.iloc[0]["answer"])
        print("Pred:")
        print(row.iloc[0]["pred"])
        #tensor_scores = df.iloc[row_num]["cosine_scores"]
        #optimal_score = df.iloc[row_num]["optimal_cosine"]

        # get optimal score as a float
        score = optimal_score.split("tensor([[")
        score_split = score[1].split("]])")
        score_split = score_split[:len(score_split)-1]
        str_score = score_split[0]
        opt_score = float(str_score)
        print("Opt", opt_score)
        # get list of scores as a float list
        str_scores = tensor_scores.split(" ")
        scores = []
        for score in str_scores:
            score = score.split("tensor([[")
            score_split = score[1].split("]])")
            score_split = score_split[:len(score_split)-1]
            str_score = score_split[0]
            score = float(str_score)
            print(score)
            scores.append(score)

        #print(scores)
        #print(len(scores))
        #print(opt_score)
        max_val = max(scores)
        max_index = scores.index(max_val)+1

        x = np.arange(1, len(scores)+1)
        print(x)
        plt.plot(x, scores, marker="o", label = f'{model}')

plt.title(f"Cosine Score per Sentence Length for {caps_title}")
plt.legend()
plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/cosine_sentence_length_plots/{genre}_cosine_scores_{row_num}_{start_token}_{end_token}.png")
    # 8 is index in dataframe not index in excel -- index in excel is 10
plt.show()  # display
