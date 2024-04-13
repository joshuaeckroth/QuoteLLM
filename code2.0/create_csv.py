import glob
import pandas as pd
"""
root_path = '../all-models-results/CSVs/'
models = ['gpt-3.5-turbo', 'gpt-4-preview-1106', 'gpt-4']

df_list = []
for model in models:
    files = glob.glob(root_path + f'{model}/*.csv3')
    for file in files:
        dir_df = pd.read_csv(file)

        filename = file.split('/')[-1]
        print(filename)
        dir_name = filename.split('-gpt')[0]
        print(dir_name)

        dir_df['category'] = [dir_name] * len(dir_df)
        print(dir_df)
        df_list.append(dir_df)

df = pd.concat(df_list)
df.to_csv('../all_data.csv')
"""
df = pd.read_csv('../all_data.csv')
print(len(df['category'].unique()))
print(len(df['file'].unique()))
print(df['category'].unique())
print(df['file'].unique())
categories = df.groupby('category')

for index, group in categories:
    print()
    print(index)
    print(group['file'].unique())
print(len(df))

# TODO: fix random text filenames
# TODO: fix nytimes filenames
# TODO: get all transcripts in github # don't push the transcripts becuase copyrighting
# TODO: get gpt-4-preivew in all_data filename issue