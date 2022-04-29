"""
Serves as a demo and enable quick wandb API calls.
This is *NOT* intended to be run as a script like this:
    $ python <myscript.py>```

But rather load this file in an interpreter, e.g.
    >> from wandb_export import *
"""

# Solve problems of nested dictionnaries

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import wandb

import sys
sys.path.insert(0, '../src')

import setup
from setup import WANDB_SETTINGS
import pandas as pd


def unnest_dataframe(df: pd.DataFrame, preprend_col_name: bool = True) -> pd.DataFrame:
    """
    Process columns of `df` having dictionnaries and put them as columns

    :param df: source dataframe
    :param preprend_col_name: whether preprend the name of the column
    """
    assert df.index.is_unique
    new_dfs = dict()
    out = df.copy()

    for col in df.columns:
        new_df = df[col].apply(pd.Series)
        # If only one column, nothing to expand
        if new_df.shape[1] > 1:
            # Rename columns
            if preprend_col_name:
                new_df.columns = [
                    f'{col}.{subcol}' for subcol in new_df.columns
                ]

                new_dfs[col] = new_df

            # Un-nest second time
            new_dfs[col] = unnest_dataframe(new_dfs[col], preprend_col_name=preprend_col_name)

    for col_to_remove, new_df in new_dfs.items():
        logging.info(f'expanding column {col_to_remove}')
        out.drop(columns=col_to_remove, inplace=True)
        out = out.join(new_df)

    return out


def parse_run_json_config(json_config: str) -> Dict[str, str]:
    json_config = json.loads(json_config)
    return {
        key: val['value'] for key, val in json_config.items()
    }


def concat_runs_history(runs: List[wandb.apis.public.Run]) -> pd.DataFrame:
    hists = []
    logging.info('extracting runs history')
    for run in runs:
        h = run.history()
        # Add id to the column
        h.insert(0, 'id', run.id)
        h.index.name = 'step'
        hists.append(h)

    df = pd.concat(hists)
    return df


def save_wandb_data(output_path, save_history=False):
    """Returns pandas.DataFrame of all runs every information that may be needed (job title, run name, url, ...)"""
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    api = wandb.Api()
    # Get list of all runs
    runs = api.runs(f"{WANDB_SETTINGS['entity']}/{WANDB_SETTINGS['project']}")

    # First get metadata, summary and config
    df, df_config = process_runs(runs)

    # Get concatenated history
    if save_history:
        df_hist = concat_runs_history(runs)

    # Save
    path_meta = output_path.joinpath('runs.csv')
    path_config = output_path.joinpath('configs.csv')
    path_hist = output_path.joinpath('histories.csv')

    logging.info(f'saving runs metadata in {str(path_meta)}')
    df.to_csv(path_meta)
    logging.info(f'saving runs configs in {str(path_config)}')
    df_config.to_csv(path_config)

    if save_history:
        logging.info(f'saving runs histories in {str(path_hist)}')
        df_hist.to_csv(path_hist)
        return df, df_config, df_hist
    return df, df_config


def process_runs(runs: List[wandb.apis.public.Run]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract runs summary and config, returns two dataframe:
        - one with id, name, entity, project and all summary metrics
        - one with id and all config parameters
    """
    def get_run_attributes(run: wandb.apis.public.Run, columns: Iterable[str]) -> Dict:
        run_data = {
            key: getattr(run, key) for key in columns
        }
        # Add sweep
        if run.sweep is not None:
            run_data['sweep_id'] = run.sweep.id
        return run_data

    columns = ['id', 'name', 'entity', 'project', 'group', 'job_type', 'state', 'url']
    # Extract runs metadata
    logging.info('extracting runs metadata')
    data = [
        get_run_attributes(run, columns) for run in runs
    ]
    df = pd.DataFrame(data).set_index('id')

    # Extract summary and config
    data_summary = []
    data_config = []
    logging.info('extracting runs summary and config')
    for run in runs:
        buffer = run.summary._json_dict
        buffer['id'] = run.id
        data_summary.append(buffer)

        buffer = parse_run_json_config(run.json_config)
        buffer['id'] = run.id
        data_config.append(buffer)
    # Merge summary with metadata
    data_summary = pd.DataFrame(data_summary).set_index('id')
    df = df.join(data_summary, on='id')
    # Build config dataframe
    df_config = pd.DataFrame(data_config).set_index('id')

    # Unnest
    df = unnest_dataframe(df, preprend_col_name=True)
    df_config = unnest_dataframe(df_config, preprend_col_name=True)

    return df, df_config


def save_run_files(runs, file_name, output_path):
    """
    Download a certain file in all runs and write in output path.
    """
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    for run in runs:
        res = run.file(file_name).download(output_path, replace=True)
        fname = f'{run.id}-{file_name}'
        fpath = output_path.joinpath(fname)
        logging.info(f'writing run {run.group}/{run.job_type}/{run.name} in {fpath}')
        shutil.move(output_path.joinpath(file_name), fpath)


if __name__ == '__main__':
    _, path = setup.get_data_paths(force_local=True)
    df, df_config = save_wandb_data(path.joinpath('wandb_data'), save_history=False)

    # api = wandb.Api()
    # runs = api.runs('mazeller/egnn-mi', filters={'group': 'evaluate'})
    # path = Path.home().joinpath('tmp/wandb')
    # save_run_files(runs, 'output.log', path)
