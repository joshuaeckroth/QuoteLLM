import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/Users/skyler/Desktop/QuoteLLM/results3.0/CSVs/"
csv_file = csv_path + "works-from-OpenAI-lawsuit-results.csv"
graph_title = "John Grisham GPT-3.5 Results"
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