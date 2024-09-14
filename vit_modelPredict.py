import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import ViTForImageClassification
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 라벨 매핑
action_labels = {
    '무언가를보다': 0, '허리굽히다': 1, '손을뻗다': 2, '몸을돌리다': 3, '핸드폰귀에대기': 4,
    '고개를돌리다': 5, '옆으로기대다': 6, '운전하다': 7, '힐끗거리다': 8, '중앙을쳐다보다': 9,
    '핸드폰쥐기': 10, '창문을열다': 11, '핸들을흔들다': 12, '꾸벅꾸벅졸다': 13, '핸들을놓치다': 14,
    '몸못가누기': 15, '무언가를마시다': 16, '무언가를쥐다': 17, '하품': 18, '중앙으로손을뻗다': 19,
    '박수치다': 20, '고개를좌우로흔들다': 21, '뺨을때리다': 22, '목을만지다': 23, '어깨를두드리다': 24,
    '허벅지두드리기': 25, '팔주무르기': 26, '눈비비기': 27, '눈깜빡이기': 28
}


# 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 2]
        image = Image.open(img_path).convert('RGB')
        action = self.data_frame.iloc[idx, 1]
        label = action_labels[action]

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 전체 데이터셋 로드
full_dataset = CustomImageDataset('dataset.csv', transform=transform)

# 10,000장의 데이터로 서브셋 생성
num_samples = 10000
_, eval_subset = random_split(full_dataset, [len(full_dataset) - num_samples, num_samples])

# 데이터로더 생성
dataloader = DataLoader(eval_subset, batch_size=64, shuffle=False)

# 사전 훈련된 모델 로드
model_path = './trained_model_subset_4'
model = ViTForImageClassification.from_pretrained(model_path, num_labels=len(action_labels))
model.eval()

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 평가
all_preds = []
all_labels = []
all_losses = []

with torch.no_grad():
    for batch in dataloader:
        inputs = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(inputs).logits
        loss = F.cross_entropy(outputs, labels)
        all_losses.append(loss.item())

        _, preds = torch.max(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 결과 출력
accuracy = accuracy_score(all_labels, all_preds)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 정밀도, 재현율, F1-스코어 등
report = classification_report(all_labels, all_preds, target_names=action_labels.keys())
print(report)

# 그래프 생성
plt.figure(figsize=(12, 6))

# 손실 그래프
plt.subplot(1, 2, 1)
plt.plot(all_losses, label='Validation Loss', marker='o')
plt.title('Validation Loss Across All Batches')
plt.xlabel('Batch Index')
plt.ylabel('Loss')
plt.legend()

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.bar(range(len(action_labels)), [accuracy_score([x], [y]) for x, y in zip(all_labels, all_preds)])
plt.title('Accuracy Per Class')
plt.xlabel('Class Index')
plt.ylabel('Accuracy')
plt.xticks(range(len(action_labels)), list(action_labels.keys()), rotation=90)
plt.tight_layout()

plt.show()
