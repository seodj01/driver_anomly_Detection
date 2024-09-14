import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from transformers import ViTForImageClassification, TrainingArguments, Trainer
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


# 랜덤 시드 설정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)  # 일관된 결과를 위한 시드 설정

# 라벨 매핑 및 데이터셋 클래스 정의
action_labels = {
    '무언가를보다': 0, '허리굽히다': 1, '손을뻗다': 2, '몸을돌리다': 3, '핸드폰귀에대기': 4,
    '고개를돌리다': 5, '옆으로기대다': 6, '운전하다': 7, '힐끗거리다': 8, '중앙을쳐다보다': 9,
    '핸드폰쥐기': 10, '창문을열다': 11, '핸들을흔들다': 12, '꾸벅꾸벅졸다': 13, '핸들을놓치다': 14,
    '몸못가누기': 15, '무언가를마시다': 16, '무언가를쥐다': 17, '하품': 18, '중앙으로손을뻗다': 19,
    '박수치다': 20, '고개를좌우로흔들다': 21, '뺨을때리다': 22, '목을만지다': 23, '어깨를두드리다': 24,
    '허벅지두드리기': 25, '팔주무르기': 26, '눈비비기': 27, '눈깜빡이기': 28
}


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

full_dataset = CustomImageDataset('dataset.csv', transform=transform)
num_subsets = 10
subset_size = len(full_dataset) // num_subsets
subsets = [Subset(full_dataset, range(i * subset_size, (i + 1) * subset_size)) for i in range(num_subsets)]

# 결과를 저장할 리스트 초기화
all_eval_losses = []
all_eval_accuracies = []

model_path = './trained_model_subset_1'
model = ViTForImageClassification.from_pretrained(model_path, num_labels=len(action_labels))

for i in range(3, num_subsets):
    subset = subsets[i]
    train_size = int(0.8 * len(subset))
    eval_size = len(subset) - train_size
    train_dataset, eval_dataset = random_split(subset, [train_size, eval_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=64)

    training_args = TrainingArguments(
        output_dir=f'./results_subset_{i}',
        num_train_epochs=3,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        logging_dir=f'./logs_subset_{i}',
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader.dataset,
        eval_dataset=eval_loader.dataset
    )

    # 훈련 시작
    trainer.train()

    # 모델 저장
    model.save_pretrained(f'./trained_model_subset_{i}')

    # 결과 데이터 추출 및 저장
    metrics = trainer.state.log_history
    eval_losses = [x['eval_loss'] for x in metrics if 'eval_loss' in x]
    eval_accuracies = [x['eval_accuracy'] for x in metrics if 'eval_accuracy' in x]
    all_eval_losses.extend(eval_losses)
    all_eval_accuracies.extend(eval_accuracies)

# 전체 서브셋 결과의 그래프 생성
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(all_eval_losses, label='Validation Loss', marker='o')
plt.title('Validation Loss Across All Subsets')
plt.xlabel('Subset Index')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(all_eval_accuracies, label='Validation Accuracy', marker='o')
plt.title('Validation Accuracy Across All Subsets')
plt.xlabel('Subset Index')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

