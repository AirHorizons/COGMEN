import json
import wandb
import numpy as np

def analyze(auth_file = '../wandb.auth', project_name = 'marg_acm2024', entity_name = 'margerc'):
    with open(auth_file, 'r') as f:
        auth_data = json.load(f)
        key = auth_data["api_key"]
        entity = auth_data["entity"]
        wandb.login(relogin=True, key=key)

        api = wandb.Api()
        runs = api.runs(f"{entity_name}/{project_name}")

        f1s = {}
        for run in runs:
            run_name = run.name
            run_type = '_'.join(run_name.split('_')[:-2])
            if run_type not in f1s:
                f1s[run_type] = []
            f1s[run_type].append(run.summary.get('best f1 (test)'))
        for run_type in f1s:
            f1s[run_type] = [item for item in f1s[run_type] if item is not None]
            print(f'{run_type}\t{np.mean(f1s[run_type])}\t{np.std(f1s[run_type])}')



if __name__ == '__main__':
    analyze()
