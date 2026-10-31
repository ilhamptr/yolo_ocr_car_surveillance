import asyncio
from ultralytics.data.converter import convert_ndjson_to_yolo

yaml_path = asyncio.run(
    convert_ndjson_to_yolo(
        "number-plate.ndjson",
        output_path="datasets"
    )
)

print("Dataset siap:", yaml_path)