import librosa # для чтения аудиофайлов, преобразования аудиоданных, извлечения признаков, обработка аудиоданных
import torch # Создание и обучение нейросети, вычисление тензоров, ускорение вычислений, модуль nn
import torch.nn as nn # создание моделей, слоев,  модулей
import torch.optim as optim # 
from transformers import BertTokenizer, BertModel # 
import json

def load_audio(file_path):
    audio, sr = librosa.load(file_path)
    audio = librosa.util.normalize(audio)  # нормализуем сигнал аудио
    return audio, sr

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__() #  вызывает медот род класса nn.Module с помощью  super, что бы осуществить инициализацию родительского класса
        self.conv1 = nn.Conv1d(1, 10, kernel_size=3) # 1 слой имеет 1 вход и 10 выходов, имеет размер ядва 3 
        self.conv2 = nn.Conv1d(10, 20, kernel_size=3) # 2 слой имеет 10 вход и 0 выходов, имеет размер ядва 3 
        self.fc1 = nn.Linear(20 * 128, 128)  # предполагая 128 временных шагов, 20 входов образуются из 20 выходов

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 20 * 128)
        x = torch.relu(self.fc1(x))
        return x
    
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def generate_text(features):
    inputs = tokenizer.encode_plus(
        'Это пример текста',  # dummy input
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )
    outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    last_hidden_state = outputs.last_hidden_state[:, 0, :]
    text = tokenizer.decode(last_hidden_state, skip_special_tokens=True)
    return text

def process_audio(file_path):
    audio, sr = load_audio(file_path)
    features = AudioCNN(audio)
    text = generate_text(features)
    return text

files = ['audio_file1.wav', 'audio_file2.mp3']  # список аудиофайлов
results = {}
for file in files:
    text = process_audio(file)
    results[file] = text

with open('results.json', 'w') as f:
    json.dump(results, f)