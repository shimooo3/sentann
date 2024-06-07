import cv2
import torch
from torchvision import transforms
from PIL import Image

import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 入力チャネル3、出力チャネル6、カーネルサイズ5
        self.pool = nn.MaxPool2d(2, 2)  # 2x2のMaxPooling
        self.conv2 = nn.Conv2d(6, 16, 5)  # 出力チャネルを16に増加
        self.fc1 = nn.Linear(16 * 13 * 13, 120)  # 全結合層
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13)  # 出力クラス数13

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # フラット化
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# モデルのロード
net = SimpleCNN()
net.load_state_dict(torch.load('./model/cnn.pth'))
net.eval()  # 推論モードへの切り替え

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 画像サイズの変更
    transforms.ToTensor(),  # テンソルに変換
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 正規化
])


def predict_frame(frame):
    """フレームに対する推論を実行する関数"""
    # OpenCVの画像をPIL形式に変換
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0)  # 前処理

    with torch.no_grad():
        outputs = net(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが開けません")
    exit()

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    if not ret:
        print("フレームが読み込めません")
        break

    # 推論を行う
    predicted_class = predict_frame(frame)
    cv2.putText(frame, f'Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # 結果を表示
    cv2.imshow('frame', frame)

    # qキーが押されたらループから抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャをリリースしてウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
