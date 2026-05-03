import shutil, os

src = r"C:\Users\htw02\project_github\lg_aimers\now_workspace\models\quantized"
dst = r"C:\Users\htw02\project_github\lg_aimers\now_workspace\models\config"

os.makedirs(dst, exist_ok=True)

for d in sorted(os.listdir(src)):
    config_path = os.path.join(src, d, "config.json")
    if os.path.isdir(os.path.join(src, d)) and os.path.isfile(config_path):
        dest_path = os.path.join(dst, f"{d}_config.json")
        shutil.copy2(config_path, dest_path)
        print(f"Copied: {d}/config.json -> {d}_config.json")

print(f"\nDone! {len(os.listdir(dst))} files in {dst}")
