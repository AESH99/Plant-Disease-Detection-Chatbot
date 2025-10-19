import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertForQuestionAnswering
import librosa
from tqdm import tqdm
import random
import json
import jiwer # A popular library for Word Error Rate (WER)

# --- Integrated Model Definitions ---
# The necessary model classes are now defined directly in this script.

class VisionModel(nn.Module):
    """A pre-trained ResNet-50 model adapted for plant disease classification."""
    def __init__(self, num_classes=38):
        super(VisionModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers except the final classification layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        num_ftrs = self.resnet.fc.in_features
        # Replace the final layer for our specific number of classes
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def get_feature_embedding(self, x):
        """Extracts features from the layer before the final classifier."""
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        return torch.flatten(x, 1)

class SpeechModel(nn.Module):
    """A CNN-LSTM model for classifying audio features (MFCCs)."""
    def __init__(self, input_features=40, hidden_dim=128, num_layers=2, num_classes=5):
        super(SpeechModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        lstm_input_size = 64 * (input_features // 4)
        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
        self.fc_embedding = nn.Sequential(nn.Linear(hidden_dim * 2, 128), nn.ReLU())

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc_head(hidden)

    def get_feature_embedding(self, x):
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc_embedding(hidden)

class FusionModel(nn.Module):
    """Combines embeddings from vision and speech models for a final prediction."""
    def __init__(self, vision_model, speech_model, num_fused_classes, vision_dim=2048, speech_dim=128):
        super(FusionModel, self).__init__()
        self.vision_model = vision_model
        self.speech_model = speech_model
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_dim + speech_dim, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_fused_classes)
        )

    def forward(self, image_input, audio_input):
        with torch.no_grad():
            vision_emb = self.vision_model.get_feature_embedding(image_input)
            speech_emb = self.speech_model.get_feature_embedding(audio_input)
        fused = torch.cat((vision_emb, speech_emb), dim=1)
        return self.fusion_mlp(fused)

# --- 1. Configuration ---
# ========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Paths ---
# IMPORTANT: Update this to the actual path of your PlantVillage dataset
DATA_DIR = "/kaggle/input/plantdisease"
# Paths for dummy data
AUDIO_DIR = "./dummy_audio_data"
TEXT_DIR = "./dummy_text_data"

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 5 # Using fewer epochs for a quick demonstration

# Create directories if they don't exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


# --- 2. Datasets and DataLoaders ---
# ===================================

# --- Vision Dataset ---
class PlantVillageVisionDataset(Dataset):
    def __init__(self, root_dir, transform=None, split='train', train_split_ratio=0.8):
        self.root_dir = root_dir
        self.transform = transform

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"The directory '{root_dir}' does not exist. Please download the PlantVillage dataset and update the DATA_DIR path.")

        all_files = []
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            # Filter out directories and only include actual image files
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath):  # Check if it's a file
                     all_files.append((fpath, class_idx))

        random.shuffle(all_files)
        split_idx = int(len(all_files) * train_split_ratio)
        if split == 'train':
            self.files = all_files[:split_idx]
        else:
            self.files = all_files[split_idx:]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path, label = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --- Dummy Audio Data Generation and Dataset ---
def generate_dummy_audio_data(num_samples=100):
    print("Generating dummy audio data...")
    metadata = []
    symptom_classes = ['Blight', 'Powdery', 'Rust', 'Curling', 'Healthy']
    for i in range(num_samples):
        file_path = os.path.join(AUDIO_DIR, f"audio_{i}.npy")
        # Simulate MFCCs of shape (n_mfcc, n_frames)
        dummy_mfcc = np.random.rand(40, 200).astype(np.float32)
        np.save(file_path, dummy_mfcc)
        label = random.choice(symptom_classes)
        metadata.append({'filepath': file_path, 'label': label})
    pd.DataFrame(metadata).to_csv(os.path.join(AUDIO_DIR, "metadata.csv"), index=False)
    return symptom_classes

class SyntheticAudioDataset(Dataset):
    def __init__(self, csv_file, root_dir, class_to_idx):
        self.metadata = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filepath = self.metadata.iloc[idx, 0]
        label_str = self.metadata.iloc[idx, 1]
        label_idx = self.class_to_idx[label_str]
        mfcc = np.load(filepath)
        # Add channel dimension: (1, n_mfcc, n_frames)
        mfcc_tensor = torch.from_numpy(mfcc).unsqueeze(0)
        return mfcc_tensor, label_idx

# --- Dummy Text QA Data Generation and Dataset ---
def generate_dummy_qa_data():
    print("Generating dummy QA data...")
    qa_data = [
        {"context": "Powdery mildew is a fungal disease that appears as white spots.", "question": "What does powdery mildew look like?", "answer": "white spots"},
        {"context": "To prevent rust, ensure good air circulation.", "question": "How do you prevent rust?", "answer": "ensure good air circulation"},
        {"context": "Blight causes sudden browning and death of tissue.", "question": "What are symptoms of blight?", "answer": "sudden browning and death of tissue"},
    ]
    with open(os.path.join(TEXT_DIR, "qa_data.json"), 'w') as f:
        json.dump(qa_data, f)

class FAQTextDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=128):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # This part is simplified for demonstration.
        start_positions = torch.tensor([10]) # Dummy start
        end_positions = torch.tensor([12])   # Dummy end
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': start_positions.flatten(),
            'end_positions': end_positions.flatten(),
        }

# --- 3. Evaluation Metrics ---
# =============================
def calculate_vision_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {"accuracy": accuracy, "f1_score": f1}

def calculate_wer(references, hypotheses):
    return jiwer.wer(references, hypotheses)

def calculate_qa_metrics(predictions, ground_truths):
    exact_match = sum(1 for p, t in zip(predictions, ground_truths) if p.strip() == t.strip())
    return {"exact_match": exact_match / len(predictions)}

# --- 4. Training Loops ---
# =========================

# --- Vision Model Training ---
def train_vision_model():
    print("\n--- Starting Vision Model Training ---")

    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        train_dataset = PlantVillageVisionDataset(DATA_DIR + "/PlantVillage", transform=transform, split='train')
        val_dataset = PlantVillageVisionDataset(DATA_DIR + "/PlantVillage", transform=transform, split='val')
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return None # Return None to signal failure

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VisionModel(num_classes=len(train_dataset.class_names)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_vision_metrics(all_labels, all_preds)
        print(f"Epoch {epoch+1} -> Val Accuracy: {metrics['accuracy']:.4f}, Val F1-Score: {metrics['f1_score']:.4f}")

    print("--- Vision Model Training Finished ---")
    return model # Return the trained model instance

# --- Speech Model Training (with Dummy Data) ---
def train_speech_model():
    print("\n--- Starting Speech Model Training (Dummy Data) ---")
    symptom_classes = generate_dummy_audio_data()
    class_to_idx = {name: i for i, name in enumerate(symptom_classes)}
    dataset = SyntheticAudioDataset(os.path.join(AUDIO_DIR, "metadata.csv"), AUDIO_DIR, class_to_idx)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SpeechModel(input_features=40, hidden_dim=128, num_layers=2, num_classes=len(symptom_classes)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (mfccs, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            mfccs, labels = mfccs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(mfccs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print("--- Speech Model Training Finished ---")
    # Note: No meaningful metrics to print for dummy data.
    return model # Return the trained model instance

# --- Text QA Model Training (with Dummy Data) ---
def train_qa_model():
    print("\n--- Starting Text QA Model Fine-Tuning (Dummy Data) ---")
    generate_dummy_qa_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(DEVICE)
    dataset = FAQTextDataset(os.path.join(TEXT_DIR, 'qa_data.json'), tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            optimizer.zero_grad()
            outputs = model(**{k: v.to(DEVICE) for k, v in batch.items()})
            loss = outputs.loss
            loss.backward()
            optimizer.step()

    print("--- Text QA Model Training Finished ---")
    # Note: No meaningful metrics to print for dummy data.
    return model, tokenizer # Return model and tokenizer

# --- Fusion Model Training (Joint Fine-tuning) ---
def train_fusion_model(vision_model, speech_model):
    print("\n--- Starting Fusion Model Fine-Tuning ---")

    fusion_model = FusionModel(vision_model, speech_model, num_fused_classes=38).to(DEVICE)
    # Freeze the backbones and only train the new MLP fusion layer
    for param in fusion_model.vision_model.parameters(): param.requires_grad = False
    for param in fusion_model.speech_model.parameters(): param.requires_grad = False

    optimizer = optim.Adam(fusion_model.fusion_mlp.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print("Simulating fusion training and evaluation with dummy inputs...")
    for epoch in range(NUM_EPOCHS):
        fusion_model.train()
        fusion_model.fusion_mlp.train() # Ensure MLP is in train mode

        # Create a dummy training batch
        dummy_image = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
        dummy_audio = torch.randn(BATCH_SIZE, 1, 40, 200).to(DEVICE)
        dummy_labels = torch.randint(0, 38, (BATCH_SIZE,)).to(DEVICE)

        optimizer.zero_grad()
        outputs = fusion_model(dummy_image, dummy_audio)
        loss = criterion(outputs, dummy_labels)
        loss.backward()
        optimizer.step()

        # --- Multi-Modal Evaluation on a dummy validation batch ---
        fusion_model.eval()
        with torch.no_grad():
            val_image = torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE)
            val_audio = torch.randn(BATCH_SIZE, 1, 40, 200).to(DEVICE)
            val_labels = torch.randint(0, 38, (BATCH_SIZE,)).to(DEVICE)
            val_outputs = fusion_model(val_image, val_audio)
            _, val_preds = torch.max(val_outputs, 1)
            metrics = calculate_vision_metrics(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            print(f"Fusion Epoch {epoch+1} -> Val Accuracy: {metrics['accuracy']:.4f}, Val F1-Score: {metrics['f1_score']:.4f}")

    print("--- Fusion Model Training Finished ---")
    return fusion_model


# --- Main Execution ---
if __name__ == "__main__":
    trained_vision_model = train_vision_model()

    if trained_vision_model:
        trained_speech_model = train_speech_model()
        trained_qa_model, _ = train_qa_model()
        trained_fusion_model = train_fusion_model(trained_vision_model, trained_speech_model)

        print("\n--- All Training and Evaluation Stages Complete! ---")
        print("Final metrics for each model are printed above.")

    else:
        print("\n--- Training Halted ---")
        print("Could not complete training because the vision dataset was not found.")
