import json

notebook_path = r"src\data\make_dataset.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

updated = False
for cell in notebook.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        source_str = "".join(source)
        if "os.makedirs('', exist_ok=True)" in source_str:
            new_source = [
                "os.makedirs('../../data/processed', exist_ok=True)\n",
                "np.save('../../data/processed/training_data.npy', normal_data.values)\n",
                "print(f\"✅ Success! Saved clean continuous dataset with shape {normal_data.shape}\")"
            ]
            cell["source"] = new_source
            updated = True
            break

if updated:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Successfully updated the notebook!")
else:
    print("Could not find the target cell in the notebook.")
