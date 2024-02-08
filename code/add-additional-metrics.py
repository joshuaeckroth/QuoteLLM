import ssl
ssl.OPENSSL_VERSION = ssl.OPENSSL_VERSION.replace("LibreSSL", "OpenSSL")
import pandas as pd
import evaluate
import re
import rich.progress
import glob

bleu = evaluate.load("bleu")
print(bleu.description)
print('----')
rouge = evaluate.load("rouge")
print(rouge.description)

with rich.progress.Progress() as progress:
    fnames = glob.glob("all-models-results/CSVs/gpt-*/*.csv")
    task_file = progress.add_task("Processing files...", total=len(fnames))
    for fname in fnames:
        progress.print(f"Processing {fname}")
        df = pd.read_csv(fname)
        task = progress.add_task("Calculating metrics...", total=len(df.index))
        df['bleu'] = 0.0
        df['rouge1'] = 0.0
        df['rouge2'] = 0.0
        df['rougeL'] = 0.0
        df['bleu_all'] = ''
        df['rouge1_all'] = ''
        df['rouge2_all'] = ''
        df['rougeL_all'] = ''
        for (idx, row) in df.iterrows():
            b_vals = []
            r1_vals = []
            r2_vals = []
            rL_vals = []
            for i in range(1, max(len(row['pred']), len(row['answer']))):
                b_val = bleu.compute(references=[''.join(row['pred'][:i])], predictions=[''.join(row['answer'][:i])])
                b_vals.append(b_val['bleu'])
                r_val = rouge.compute(references=[''.join(row['pred'][:i])], predictions=[''.join(row['answer'][:i])])
                r1_vals.append(r_val['rouge1'])
                r2_vals.append(r_val['rouge2'])
                rL_vals.append(r_val['rougeL'])
            df.at[idx, 'bleu_all'] = ','.join(map(lambda v: "%.2f" % v, b_vals))
            df.at[idx, 'rouge1_all'] = ','.join(map(lambda v: "%.2f" % v, r1_vals))
            df.at[idx, 'rouge2_all'] = ','.join(map(lambda v: "%.2f" % v, r2_vals))
            df.at[idx, 'rougeL_all'] = ','.join(map(lambda v: "%.2f" % v, rL_vals))
            df.at[idx, 'bleu'] = max(b_vals)
            df.at[idx, 'rouge1'] = max(r1_vals)
            df.at[idx, 'rouge2'] = max(r2_vals)
            df.at[idx, 'rougeL'] = max(rL_vals)
            progress.update(task, advance=1)

        df.to_csv(re.sub(r'\.csv$', '-additional-metrics.csv2', fname), index=False)
        progress.update(task_file, advance=1)
