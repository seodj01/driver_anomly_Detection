import os
import json
import pandas as pd

def create_csv(label_dir, image_dir, output_csv):
    data = []

    for root, dirs, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                # 경로에서 video_id와 scene_id 추출
                parts = root.split(os.sep)  # Windows 환경에서는 경로 구분자가 \ 이므로 os.sep 사용
                video_id = parts[-3]  # 예: SGA2100300
                scene_id = parts[-2]  # 예: SGA2100300S00190

                for scene in json_data['scene']['data']:
                    img_name = scene['img_name']
                    img_folder = os.path.join(image_dir, video_id, scene_id, "img")
                    full_image_path = os.path.join(img_folder, img_name)
                    full_image_path = full_image_path.replace("\\", "/")  # 유닉스 스타일 경로로 변경

                    record = {
                        'img_name': img_name,
                        'action': scene['occupant'][0]['action'],
                        'img_path': full_image_path
                    }
                    data.append(record)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, encoding='utf-8')

label_dir = './dataset/labelDataset/TL1'
image_dir = './dataset/sourceDataset/TS1'
output_csv = './dataset.csv'
create_csv(label_dir, image_dir, output_csv)
