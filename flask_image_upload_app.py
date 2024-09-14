from flask import Flask, request, render_template, jsonify, send_from_directory
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import uuid

app = Flask(__name__)

# 이미지 업로드 폴더 경로
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
probability_threshold = 0.5  # 확률 기준

# 모델 정의 및 로드
model = models.resnet18()
num_classes = 29  # 실제 학습한 클래스 수로 변경
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

checkpoint = torch.load('./Resnet_final_checkpoint/subset_10_epoch_5_checkpoint.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 클래스 이름 매핑 및 그룹 설정
class_names = {
    0: '무언가를보다', 1: '허리굽히다', 2: '손을뻗다', 3: '몸을돌리다', 4: '핸드폰귀에대기', 
    5: '고개를돌리다', 6: '옆으로기대다', 7: '운전하다', 8: '힐끗거리다', 9: '중앙을쳐다보다', 
    10: '핸드폰쥐기', 11: '창문을열다', 12: '핸들을흔들다', 13: '꾸벅꾸벅졸다', 14: '핸들을놓치다', 
    15: '몸못가누기', 16: '무언가를마시다', 17: '무언가를쥐다', 18: '하품', 19: '중앙으로손을뻗다', 
    20: '박수치다', 21: '고개를좌우로흔들다', 22: '뺨을때리다', 23: '목을만지다', 24: '어깨를두드리다', 
    25: '허벅지두드리기', 26: '팔주무르기', 27: '눈비비기', 28: '눈깜빡이기'
}

# 그룹 정의
group_1 = {0, 19, 10, 8}  # "운전에 집중하세요"
group_2 = {14, 22, 2, 20, 4, 16, 17, 21, 27}  # "딴짓하지 마세요"
group_3 = {7, 9, 11, 26, 12}  # "잘하고 있어요"
group_4 = {18, 13, 15, 23, 24, 25, 28, 1, 5, 6}  # "졸지 마세요"

def get_warning_message(predicted_class):
    if predicted_class in group_1:
        return "운전에 집중하세요!"
    elif predicted_class in group_2:
        return "딴짓하지 마세요!"
    elif predicted_class in group_3:
        return "잘하고 있어요!"
    elif predicted_class in group_4:
        return "졸지 마세요!"
    else:
        return "Unknown behavior"

def predict(image):
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
    
    probabilities = F.softmax(output, dim=1)
    max_prob, predicted = torch.max(probabilities, 1)
    
    if max_prob.item() >= probability_threshold:
        predicted_class = predicted.item()
        prediction = class_names.get(predicted_class, "Unknown")
        warning_message = get_warning_message(predicted_class)
        danger = predicted_class in group_4  # 졸음 그룹일 경우 위험 경고
    else:
        prediction = "Uncertain"
        warning_message = "Uncertain behavior"
        danger = False
    
    return prediction, max_prob.item(), warning_message, danger

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # 이미지 저장
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 이미지 로드 및 예측
        img = Image.open(filepath).convert('RGB')
        prediction, probability, warning_message, danger = predict(img)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'warning_message': warning_message,
            'danger': danger,
            'image_url': f'/uploads/{filename}'  # 이미지 URL 반환
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
