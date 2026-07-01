from pathlib import Path

file_dir = Path(__file__)
root_folder = file_dir.parent.parent

print("Creating DATASETS folders... ")
Path(root_folder / "DATASETS").mkdir(parents=True, exist_ok=True)
Path(root_folder / "DATASETS" / "Train").mkdir(parents=True, exist_ok=True)
Path(root_folder / "DATASETS" / "Test").mkdir(parents=True, exist_ok=True)

print("Done")