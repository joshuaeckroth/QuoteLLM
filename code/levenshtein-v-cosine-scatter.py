import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

filenames = glob.glob('/Users/skyler/Desktop/QuoteLLM/results3.0/CSVs/*')
for filename in filenames:
    df = pd.read_csv(filename)
    file = filename.split("/")[-1]
    genre = file.split("-results.csv")[0] #Sci-Fi

    title = genre.split("-")
    spaced_title = " ".join(title)
    caps_title = spaced_title.title()

# df = pd.read_csv("/Users/skyler/Desktop/QuoteLLM/results2.0/CSVs/published-post-model-results.csv")

    levenshtein_distances = df["levenshtein_distance"]
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
    x = levenshtein_distances
    plt.xlabel("Levenshtein Distance")
    plt.ylabel("Optimal Cosine Score")
    plt.title(f"Levenshtein vs. Optimal Cosine for {caps_title}")
    plt.scatter(x, y)  # Plot the chart
    plt.gca().invert_yaxis()
    plt.savefig(f"/Users/skyler/Desktop/QuoteLLM/results3.0/visualization/cosine_levenshtein_scatter_plots/{genre}_levenshtein_v_cosine.png")
    plt.show()  # display
