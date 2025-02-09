import json
import os


INPUT_ANNOTATIONS_FILE = "sky_classification_export/sky_classification_export.json" 
OUTPUT_COCO_FILE = "sky_classification_coco.json"  

# Define category mapping
categories = [
    {"id": 1, "name": "commercial"},
    {"id": 2, "name": "nature"},
    {"id": 3, "name": "other"},
    {"id": 4, "name": "residential"},
]

# Create COCO JSON structure
coco_format = {
    "images": [],
    "annotations": [],
    "categories": categories
}


with open(INPUT_ANNOTATIONS_FILE, "r") as f:
    dataset = json.load(f)


image_id = 0
annotation_id = 0
for entry in dataset:
    image_path = entry["image"]

    if "choice" not in entry:
        continue  

    category_name = entry["choice"]

    category_id = next((cat["id"] for cat in categories if cat["name"] == category_name), None)
    
    if category_id is None:
        continue  


    coco_format["images"].append({
        "id": image_id,
        "file_name": image_path,
        "width": 640,  
        "height": 640  
    })

    coco_format["annotations"].append({
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
    })

    image_id += 1
    annotation_id += 1

with open(OUTPUT_COCO_FILE, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO dataset saved to {OUTPUT_COCO_FILE}")
