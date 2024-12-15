import os 
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_key(key):
    parts = key.split('_')
    d = {
        'Model': parts[0],
        'Epochs': int(parts[1][1:]),
        'Batch Size': int(parts[2][1:])
    }
    if len(parts) == 3:
        d['Learning Rate'] = "1e-3"
        d["Optimizer"] = "Adam"
    elif len(parts) == 4:
        if parts[3][:2] == 'lr':
            lr = parts[3][2:] if parts[3][2:] != '1' else "3"
            d['Learning Rate'] = "1e-" + lr
            d["Optimizer"] = "Adam"
        else:
            d['Learning Rate'] = "1e-3"
            d["Optimizer"] = parts[3]
    else:
        lr = parts[3][2:] if parts[3][2:] != '1' else "3"
        d['Learning Rate'] = "1e-" + lr
        d["Optimizer"] = parts[4]
    
    s = f"E:{d['Epochs']}, B:{d['Batch Size']}, LR:{d['Learning Rate']}, {d['Optimizer']}"

    return s

dir = "../Hyp_Tuning/LeNet/"

data = {}

for root, dirnames, files in os.walk(dir):
    for dir in dirnames:
        path = os.path.join(root, dir, "classification_report.txt")

        try:
            with open(path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
            data[dir] = last_line
        except:
            pass


rows = []
for key, value in data.items():
    match = re.search(r'weighted avg\s+(\d\.\d+)\s+(\d\.\d+)\s+(\d\.\d+)', value)
    if match:
        precision, recall, f1 = map(float, match.groups())
        rows.append({
            'Model': parse_key(key),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

df = pd.DataFrame(rows)
df = df.sort_values(by='F1-Score', ascending=True)

plt.figure(figsize=(12, 6))
plt.plot(df['Model'], df['Precision'], marker='o', label='Precision', alpha=0.4)
plt.plot(df['Model'], df['Recall'], marker='o', label='Recall', alpha=0.4)
plt.plot(df['Model'], df['F1-Score'], marker='o', label='F1-Score', alpha=1)
plt.xticks(rotation=90, fontsize=8)
plt.title('Performance Metrics by Model')
plt.xlabel('Model')
plt.ylabel('Metrics')
plt.legend()
plt.tight_layout()
plt.savefig('performance_metrics.png')
        