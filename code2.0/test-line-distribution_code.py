import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from astropy.table import Table

def setup_line_distribution(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)



def line_distribution(file, metric, filepath):
    # df = pd.read_csv(file)
    plt.figure(figsize=(8, 6))
    df = Table.read('../all-models-results/visualization/ecdfs/bleu-ecdf-table.tex').to_pandas()
    print(df)
    print(df.columns)
    models = ['gpt-3.5-turbo', 'gpt-4-1106-preview', 'gpt-4'] # TODO: make a tuple?
    sub_location = [4, 6, 8]
    pos = 0
    for model in models:
        ax = plt.subplot(sub_location[pos], 1, 4)
        pos += 1
        setup_line_distribution(ax)
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.text(0.0, 0.1, model, fontsize=8, transform=ax.transAxes)
        model_vals = df[f'{model}'].tolist()
        categories = df['Category'].tolist()
        index = 0
        for val in model_vals:
            plt.plot(val, 0, '|' , ms = 50, label= df.loc[index, 'Category']) # TODO: make sure all colors line up across models
            # plt.text(val, 1, df.loc[index, 'Category'], fontsize=6, horizontalalignment='right', rotation=90)
            index += 1
    plt.title('ECDF Scores per Model and Category')
    plt.legend() # TODO: fix legend location, label, and marker
    plt.savefig('../line_distribution.png')
    plt.show()
    # TODO: plot spacing in figure size


if __name__ == '__main__':

    file = '../all_data.csv'
    df = pd.read_csv(file)  # put under CSVs

    models = df['model'].unique()
    categories = df['category'].unique()
    # metric = ['levenshtein', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'optimal_cosine']
    metric = ['bleu']

    for metric in metric:  # TODO: make all ecdf scores across the 3 models go to the ecdf csv in the ecdf method
        line_distribution(f'../{metric}_ecdf_data.csv', metric, f'../{metric}_line_distribution.png')  #
filepath = '../{model}/{metric}/ecdf_data.csv'
pd.read_csv(f'{filepath}')


