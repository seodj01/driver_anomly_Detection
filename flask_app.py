import base64
import io

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, emit
import torch
from transformers import ViTForImageClassification
import torchvision.transforms as transforms
import cv2
from PIL import Image
import winsound
import os
import torch.nn.functional as F


app = Flask(__name__)
socketio = SocketIO(app)

# 모델 로드
model_path = './trained_model_subset_4'
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

# 액션 라벨과 그룹 정의
action_labels = {
    0: '무언가를보다', 1: '허리굽히다', 2: '손을뻗다', 3: '몸을돌리다', 4: '핸드폰귀에대기',
    5: '고개를돌리다', 6: '옆으로기대다', 7: '운전하다', 8: '힐끗거리다', 9: '중앙을쳐다보다',
    10: '핸드폰쥐기', 11: '창문을열다', 12: '핸들을흔들다', 13: '꾸벅꾸벅졸다', 14: '핸들을놓치다',
    15: '몸못가누기', 16: '무언가를마시다', 17: '무언가를쥐다', 18: '하품', 19: '중앙으로손을뻗다',
    20: '박수치다', 21: '고개를좌우로흔들다', 22: '뺨을때리다', 23: '목을만지다', 24: '어깨를두드리다',
    25: '허벅지두드리기', 26: '팔주무르기', 27: '눈비비기', 28: '눈깜빡이기'
}

group_labels = {
    'group1': [0, 19, 10, 8],
    'group2': [14, 22, 2, 20, 4, 16, 17, 21, 27],
    'group3': [7, 9, 11, 26, 12],
    'group4': [18, 13, 15, 23, 24, 25, 28, 1, 5, 6]
}

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 그룹에 따른 메시지와 스타일 설정
def get_label_and_style(predicted_class_idx):
    label = action_labels[predicted_class_idx]
    if predicted_class_idx in group_labels['group1']:
        message = "운전에 집중하세요"
        style = "color: blue;"
    elif predicted_class_idx in group_labels['group2']:
        message = "딴짓하지 마세요"
        style = "color: purple;"
    elif predicted_class_idx in group_labels['group3']:
        message = "잘하고 있어요"
        style = "color: green;"
    elif predicted_class_idx in group_labels['group4']:
        message = "졸지마세요"
        style = "color: red;"
        winsound.Beep(1000, 500)
    else:
        message = "Unknown"
        style = "color: black;"

    return f"{message}: {label}", style


def predict_image(image):
    # 이미지를 RGB로 변환 (알파 채널이 있는 경우 대비)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # 이미지를 전처리
    inputs = transform(image).unsqueeze(0)  # 배치 차원 추가
    outputs = model(inputs)
    logits = outputs.logits

    # 소프트맥스를 사용하여 확률 계산
    probabilities = F.softmax(logits, dim=-1)
    predicted_class_idx = probabilities.argmax(-1).item()
    highest_prob = probabilities[0][predicted_class_idx].item()

    # 예측된 클래스 라벨과 스타일 가져오기
    label, style = get_label_and_style(predicted_class_idx)

    return label, style, highest_prob

# 카메라에서 프레임 읽기
def get_frame():
    camera = cv2.VideoCapture(0)  # 카메라 장치 열기
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 프레임을 PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 이미지 전처리
            inputs = transform(pil_image).unsqueeze(0)  # 배치 차원 추가
            outputs = model(inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

            # 예측된 클래스 라벨과 스타일 추가
            label, style = get_label_and_style(predicted_class_idx)

            # 서버에서 클라이언트로 라벨과 스타일 전송
            socketio.emit('prediction', {'label': label, 'style': style})

            # 프레임을 byte 형태로 변환
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# 웹페이지 루트 라우트
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', label="파일이 업로드되지 않았습니다.")

        file = request.files['file']

        if file.filename == '':
            return render_template('upload.html', label="파일이 선택되지 않았습니다.")

        if file and file.filename.endswith(('png', 'jpg', 'jpeg')):
            try:
                # 파일을 PIL 이미지로 변환
                image = Image.open(file)

                # 예측 수행
                label, style, highest_prob = predict_image(image)

                # 이미지를 다시 클라이언트로 보내기 위해 base64로 인코딩
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                img_data = base64.b64encode(img_io.getvalue()).decode()

                return render_template('upload.html', label=label, style=style, img_data=img_data,
                                       highest_prob=highest_prob)
            except Exception as e:
                return render_template('upload.html', label=f"예측 중 오류 발생: {str(e)}")

    return render_template('upload.html', label="파일을 업로드하세요.")

# 비디오 피드 라우트
@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
