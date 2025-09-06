import json
import pickle
from pathlib import Path

# Pfade anpassen
model_path = Path("models/TTS/de_DE-glados-high")
json_path = model_path.with_name(model_path.name + ".json")
pkl_path = model_path.parent / "phoneme_to_id_german.pkl"

# JSON laden
with open(json_path, encoding="utf-8") as f:
    config = json.load(f)

# phoneme map rausholen
raw_map = config.get("phoneme_id_map") or config.get("phoneme_to_id") or {}
id_map = {}
for k, v in raw_map.items():
    if isinstance(v, list):
        id_map[k] = [int(x) for x in v]
    else:
        id_map[k] = [int(v)]

# pickle speichern
with open(pkl_path, "wb") as f:
    pickle.dump(id_map, f)

print(f"✅ phoneme_to_id.pkl gespeichert unter: {pkl_path}")
