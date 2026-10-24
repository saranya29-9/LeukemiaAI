from pathlib import Path
import zipfile
import random

PROJECT_DIR = Path(__file__).parent
ZIP_PATH = PROJECT_DIR / "data" / "Leukemia_Peripheral_Smear_HPF_Dataset.zip"
OUTPUT_DIR = PROJECT_DIR / "data" / "sample_200"

IMAGES_PER_CLASS = 200
SEED = 42

random.seed(SEED)

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    all_files = zip_ref.namelist()

    class_names = [
        "Acute Lymphoblastic Leukemia",
        "Acute Myeloid Leukemia",
        "Chronic Lymphocytic Leukemia",
        "Chronic Myeloid Leukemia",
        "Normal & Reactive",
    ]

    for class_name in class_names:
        class_prefix = (
            "Leukemia_Peripheral_Smear_HPF_Dataset/"
            + class_name
            + "/"
        )

        image_files = [
            file for file in all_files
            if file.startswith(class_prefix)
            and file.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        selected_files = random.sample(
            image_files,
            min(IMAGES_PER_CLASS, len(image_files))
        )

        output_class_dir = OUTPUT_DIR / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExtracting {class_name}: {len(selected_files)} images")

        for file in selected_files:
            filename = Path(file).name
            output_path = output_class_dir / filename

            if not output_path.exists():
                with zip_ref.open(file) as source:
                    with open(output_path, "wb") as destination:
                        destination.write(source.read())

print("\nSample dataset preparation completed!")
print("Output folder:", OUTPUT_DIR)