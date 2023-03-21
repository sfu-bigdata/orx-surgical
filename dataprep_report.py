
"""Generate a dataprep report via `create_report`
Usage:
```
$ python dataprep_report.py features.parquet report_01 'Features from Task and Form completion'
```
Creates a report (titled 'Features from Task and Form completion') located ub `report_01.html`of the dataframe in the parquet file `features.parquet`.
"""


from dataprep.eda import create_report
import pandas as pd
import sys
import cms_preprocess

def main(parquet_file, output_file, title:str):
    
    # features_saved = pd.read_parquet(parquet_file)
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = cms_preprocess.get_aov(ben_path=cms_preprocess.ben_path, ip_path=cms_preprocess.ip_path,
         pde_path=cms_preprocess.pde_path, dx_path=cms_preprocess.dx_path, pcs_path=cms_preprocess.pcs_path, ben_cols=cms_preprocess.ben_cols, ip_cols=cms_preprocess.ip_cols,
        pde_cols=cms_preprocess.pde_cols, start_year=2008, end_year=2010, random_state = 42, col_num=6)
    
    print("Loaded")
    features_saved = pd.DataFrame(x_train)
    print("Report")
    create_report(features_saved, title=title).save(output_file)


if __name__ == "__main__":

    input_parquet = sys.argv[1]
    output_file = sys.argv[2]
    title = sys.argv[3]

    print(f"\n{input_parquet=}, {output_file=}, {title=}\n")
    
    main(input_parquet, output_file, title=title)
    
    print()
    