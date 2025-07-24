import torch



train_datasets = torch.rand(100,5)
y  = train_datasets.argmax(dim=1)

test_datasets = torch.rand(100,5)
y_test  = test_datasets.argmax(dim=1)

print("y:",y)

class mymodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 80)
        self.linear2 = torch.nn.Linear(80, 5)
    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(x)
        return x


def main():
    Epoch = 300
    model = mymodel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("model.parameters",model.state_dict())
    for epoch in range(Epoch):
        y_pred = model(train_datasets)
        loss = torch.nn.functional.cross_entropy(y_pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        accuracy = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
        print(f"epoch:{epoch+1}/{Epoch},loss:{loss},accuracy:{accuracy}")

        if epoch % 10 == 0:
            y_pred = model(test_datasets)
            loss = torch.nn.functional.cross_entropy(y_pred,y_test)
            accuracy = (y_pred.argmax(dim=1) == y_test).sum().item() / len(y_test)
            print(f"epoch:{epoch+1}/{Epoch},test_loss:{loss},test_accuracy:{accuracy}")


    torch.save(model.state_dict(), "model.pt")

if __name__ == '__main__':
    main()



