import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, gridspec
from astropy.table import Table

def setup_line_distribution(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.patch.set_alpha(0.0)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)




def line_distribution(file, metric, filepath):
    # df = pd.read_csv(file)
    plt.figure(figsize=(8, 6))
    fig, axs = plt.subplots(6, 1, figsize=(8, 6))
    df = Table.read('../all-models-results/visualization/ecdfs/bleu-ecdf-table.tex').to_pandas()
    models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview']
    pos = 0
    shift_left = ['Copyright Lawsuit Works', 'Fantasy', 'Famous Quote']

    for model in models:
        # Plot points on each subplot
        setup_line_distribution(axs[pos])
        if pos == 2:
            axs[pos].xaxis.set_major_locator(ticker.AutoLocator())
            axs[pos].xaxis.set_minor_locator(ticker.AutoMinorLocator())
            axs[pos].text(0.0, 0.1, model, fontsize=8, transform=axs[pos].transAxes)
        else :
            axs[pos].xaxis.set_major_locator(ticker.NullLocator())
            axs[pos].text(0.0, 0.1, model, fontsize=8, transform=axs[pos].transAxes)

        model_vals = df[f'{model}'].tolist()

        index = 0
        for val in model_vals:
            # plt.plot(val, 0, '|' , ms = 75, label= df.loc[index, 'Category'])
            axs[pos].plot(val, 0, '|', ms=75, label=df.loc[index, 'Category'])
            print(df.loc[index, 'Category'])
            if pos == 2:
                if df.loc[index, 'Category'] in (shift_left):
                    axs[pos].text(val-0.02, -0.15, df.loc[index, 'Category'], fontsize=10, verticalalignment='top', rotation=270)
                else:
                    axs[pos].text(val, -0.15, df.loc[index, 'Category'], fontsize=10, verticalalignment='top', rotation=270)
            index+=1
        pos += 1

    # Create value and category lists
    categories = df.Category.unique()
    gpt_1 = df['gpt-3.5-turbo'].tolist()
    gpt_2 = df['gpt-4'].tolist()
    gpt_3 = df['gpt-4-1106-preview'].tolist()

    # Connect categories across subplots with lines
    for i, cat in enumerate(categories):
        axs[0].plot([gpt_1[i], gpt_2[i]], [i, i], color='black')
        axs[0].plot([gpt_2[i], gpt_3[i]], [i, i], color='black')

    # Hide extra rows
    for i in range(3, 6):
        axs[i].axis('off')

    plt.title('ECDF Scores per Model and Category')
    # plt.legend(bbox_to_anchor=(0.5, -1.25), loc='lower center', markerscale=0.1)
    plt.savefig('../line_distribution.png', dpi = 300)
    plt.show()

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

# make one plot and use values as coordinates
# draw hand draw line segments connecting each
# don't use colors use labels
# then do little sections at bottom showing groups
