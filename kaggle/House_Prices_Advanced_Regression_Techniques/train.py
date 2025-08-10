import torch
import matplotlib.pyplot as plt
from data_read import get_data
from  MyNet import MyNet
import os

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"{device}")

def train_model(epochs=200,lr=1e-2,batch_size=32):
    _, _, train_X, _, _, train_loader = get_data(batch_size)
    in_channel = train_X.shape[1]
    model=MyNet(in_channel).to(device)
    loss_fn=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    loss_list=[]
    for epoch in range(1,epochs+1):
        model.train()
        epoch_loss=0.0
        for batch_X,batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            pred = model(batch_X)
            loss = loss_fn(pred, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(batch_X)

        epoch_loss /= len(train_loader)
        loss_list.append(epoch_loss)
        if epoch % 20 == 0 or epoch == 1:
            print(f"[train] Epoch {epoch:3d} / {epochs} | loss = {epoch_loss:.4f}")

    os.makedirs("./img", exist_ok=True)
    plt.figure()
    plt.plot(range(1, epochs + 1), loss_list)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("./img/training_loss.png")
    plt.show()

    os.makedirs("./model", exist_ok=True)
    torch.save(model.state_dict(), "./model/model.pth")
    print("[train] 模型已保存到 ./model/model.pth")

if __name__ == "__main__":
    train_model()