import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt


#model_list = ["gpt-3.5-turbo", "gpt-4-1106-preview", "gpt-4"]
model_list = ["gpt-3.5-turbo"]
prompt_type = "random-author"
for model in model_list:

    #filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{model}/*.csv3')
    filenames = glob.glob(f'/Users/skyler/Desktop/QuoteLLM/all-models-results/CSVs/{prompt_type}-*')
    for filename in filenames:
        df = pd.read_csv(filename)
        file = filename.split("/")[-1]
        genre1 = file.split("-gpt")[0] #Sci-Fi # no-author-fantasy
        genre =  genre1.split(f"{prompt_type}-")[-1] # fantasy

        title = genre.split("-")
        spaced_title = " ".join(title)
        caps_title = spaced_title.title()

    # df = pd.read_csv("/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/published-post-model-results.csv")
        metric = "levenshtein_distance"
        metric_scores = df[f"{metric}"]
        optimal_scores = df["optimal_cosine"]

        """
        # get optimal score as a float
        score = optimal_score.split("tensor([[")
        score_split = score[1].split("]])")
        score_split = score_split[:len(score_split)-1]
        str_score = score_split[0]
        opt_score = float(str_score)
        print("Opt", opt_score)
        """

        opt_scores = []
        for score in optimal_scores:
            score = score.split("tensor([[")
            score_split = score[1].split("]])")
            score_split = score_split[:len(score_split)-1]
            str_score = score_split[0]
            score = float(str_score)
            print(score)
            opt_scores.append(score)

        print(opt_scores)
        y = opt_scores
        x = metric_scores
        plt.figure(figsize=(15, 6))
        plt.xlabel(f"{metric}") #plt.xlabel(f"{metric.upper()} Score")
        plt.ylabel("Optimal Cosine Score")
        plt.title(f"{metric} vs. Optimal Cosine for {caps_title} with {model}")
        plt.scatter(x, y)  # Plot the chart
        plt.gca().invert_yaxis()
        plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/all-models-results/visualization/{prompt_type}-visualization/{genre}_{metric}_v_cosine-{prompt_type}.png")
        plt.show()  # display
