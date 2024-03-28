import subprocess
from pathlib import Path

import pandas as pd


def lazy_add_response_property_to_object_level(object_df, object_cfg, response_property_name):
    if response_property_name not in object_df.columns:
        # we need to run the property extraction command
        run_property_extraction_command = f"python -m evals.run_property_extraction response_property={response_property_name} dir={object_cfg.exp_dir}"
        print(
            f"Response property {response_property_name} not in object level dataframe. Running property extraction with `{run_property_extraction_command}`."
        )
        # run the shell command
        main_path = Path(__file__).parent.parent.parent  # get the repo directory
        subprocess.run(run_property_extraction_command, shell=True, check=True, cwd=main_path)
        # now the file in the exp_dir should have the response property
        updated_object_df = pd.read_csv(str(main_path) + "/" + object_cfg.exp_dir + "/" + f"data{object_cfg.seed}.csv")
        # load in the new column from the object level dataframe into the current one by joining on string
        object_df = pd.merge(object_df, updated_object_df[["string", response_property_name]], on="string")
        print(f"Loaded response property {response_property_name} from object level dataframe.")
    return object_df
