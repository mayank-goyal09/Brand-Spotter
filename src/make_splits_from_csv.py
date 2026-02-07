import shutil
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "data" / "logos_small"

train_dir = BASE / "train"
val_dir   = BASE / "val"
youtube_src = BASE / "youtube"   # currently outside train/val

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def move_folder_contents(src: pathlib.Path, dst: pathlib.Path):
    ensure_dir(dst)
    if not src.exists():
        print(f"[skip] {src} not found")
        return
    for f in src.iterdir():
        if f.is_file():
            shutil.move(str(f), str(dst / f.name))

# 1) If youtube exists at root, move its images into train/youtube
train_youtube = train_dir / "youtube"
val_youtube   = val_dir / "youtube"

ensure_dir(train_youtube)
ensure_dir(val_youtube)

# move all youtube images to train/youtube (simple version)
move_folder_contents(youtube_src, train_youtube)

# 2) Ensure val has the same class folders as train
for cls_folder in train_dir.iterdir():
    if cls_folder.is_dir():
        ensure_dir(val_dir / cls_folder.name)

print("Done. Verify folders:")
print("Train classes:", sorted([p.name for p in train_dir.iterdir() if p.is_dir()]))
print("Val classes:", sorted([p.name for p in val_dir.iterdir() if p.is_dir()]))
