import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import time

DATA_DIR='custom_dataset' #делал фото своих одногруппников, по 100 фрир уставишими и 100 бодрыми. 6 человек, считая меня). Стало работать лучше, чем с ddd
BATCH_SIZE=16
EPOCHS=5

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except FileNotFoundError:
        print("Папка не найдена")
        return

    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Для тренировки: {train_size}, для проверки: {val_size}\n")
    model=models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

    # замораживаем тело
    for param in model.parameters():
        param.requires_grad = False

    # новая бошка + два класса в ней
    num_features=model.heads.head.in_features
    model.heads.head=nn.Linear(num_features, 2)
    model=model.to(device)

    # 4. Учитель и оптимизатор
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.heads.head.parameters(), lr=0.001)

    # 5. Цикл обучения
    print("Начинаем обучение")
    start_time = time.time() #30 минут на моем ноуте)

    for epoch in range(EPOCHS):
        model.train()
        running_loss=0.0
        correct=0
        total=0

        for images, labels in train_loader:
            images, labels=images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()
            _, predicted=torch.max(outputs.data, 1)
            total+=labels.size(0)
            correct+=(predicted == labels).sum().item()

        epoch_loss=running_loss / len(train_loader)
        epoch_acc=100*correct / total
        
        print(f"Эпоха: {epoch+1} Ошибка: {epoch_loss:.4f} Accuracy: {epoch_acc:.1f}%")

    print("Валидация")
    model.eval()
    val_correct=0
    val_total=0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs=model(images)
            _, predicted=torch.max(outputs.data, 1)
            val_total+=labels.size(0)
            val_correct+=(predicted==labels).sum().item()

    val_acc = 100*val_correct / val_total
    print(f"Итог Accuracy: {val_acc:.1f}%")

    torch.save(model.state_dict(), 'tivit_weights.pth')
    
    elapsed=(time.time()-start_time)/60
    print(f"Обучение заняло {elapsed:.1f}  минут.")

if __name__ == "__main__":
    main()