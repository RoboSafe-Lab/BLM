import argparse
import pandas as pd
from pathlib import Path


def main(args):
    expected_columns = [
        'scene_name', 'num_agents', 'min_ttc', 'off-road_rate', 'collision_rate',
        'rss_lon', 'rss_lat', 'jsd_vel', 'jsd_acc', 'jsd_jerk', 'fdd'
    ]

    folder_path = Path(args.inputs_dir)
    if not folder_path.exists():
        raise FileNotFoundError(f"folder not exist: {folder_path}")
    csv_files = list(folder_path.glob("*.csv"))
    print(f"find {len(csv_files)} CSVs")

    dataframes = {}
    scene_name_sets = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # check file valid
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"file {csv_file.name} invalid")

        dataframes[csv_file] = df
        scene_names = set(df['scene_name'].dropna().unique())
        scene_name_sets.append(scene_names)

    # find common scene_name
    common_scenes = set.intersection(*scene_name_sets) if scene_name_sets else set()

    print(f"find {len(common_scenes)} common scenes")

    for csv_file, df in dataframes.items():
        filtered_df = df[df['scene_name'].isin(common_scenes)]
        filtered_df = filtered_df.sort_values('scene_name')
        output_file = f"{args.outputs_dir}/{csv_file.stem}_common.csv"

        filtered_df.to_csv(output_file, index=False)

    print("saved all data!")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputs_dir",
        type=str,
        required=True,
        help="A folder of all CSVs",
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="A output folder"
    )

    args = parser.parse_args()

    main(args)