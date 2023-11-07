import os
import pandas as pd

from yz_ds_projects.california_traffic_collision_database_2023.config import RAW_DATA_PATH, TYPE_LIST, DATA_PATH


def merge_type_df():
    for t in TYPE_LIST:
        if os.path.exists(os.path.join(DATA_PATH, '{}.csv').format(t)):
            print('{} already exists! SKIP!'.format(os.path.join(DATA_PATH, '{}.csv').format(t)))
        type_df_list = []
        for d in os.listdir(RAW_DATA_PATH):
            if os.path.isdir(os.path.join(RAW_DATA_PATH, d)):
                file_name = '{}_{}.txt'.format(d, t)
                file_path = os.path.join(RAW_DATA_PATH, d, file_name)
                print('Reading file: {}'.format(file_path))
                df = pd.read_csv(file_path, sep=',', on_bad_lines='skip')
                df['CASE_ID'] = df['CASE_ID'].astype(int)
                type_df_list.append(df)
        all_date_df = pd.concat(type_df_list)
        all_date_df.drop_duplicates(subset=['CASE_ID'], inplace=True)
        all_date_df.to_csv(os.path.join(DATA_PATH, '{}.csv'.format(t)), index=False)
        print('Saved {}'.format(os.path.join(DATA_PATH, '{}.csv'.format(t))))
    print('Merge done.')


if __name__ == "__main__":
    merge_type_df()
