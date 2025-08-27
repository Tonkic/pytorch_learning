
import torch
from src.data.transformer_data import zidian_y, loader, zidian_xr, zidian_yr
from src.utils.transformer_mask import mask_pad, mask_tril
from src.models.transformer_model import Transformer

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 预测函数
def predict(x):
    # x = [1, 50]
    model.eval()
    x = x.to(device)
    # [1, 1, 50, 50]
    mask_pad_x = mask_pad(x)
    # 初始化输出,这个是固定值
    target = [zidian_y['<SOS>']] + [zidian_y['<PAD>']] * 49
    target = torch.LongTensor(target).unsqueeze(0).to(device)
    # x编码,添加位置信息
    x = model.embed_x(x)
    # 编码层计算,维度不变
    x = model.encoder(x, mask_pad_x)
    # 遍历生成第1个词到第49个词
    for i in range(49):
        y = target
        mask_tril_y = mask_tril(y)
        y = model.embed_y(y)
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        out = model.fc_out(y)
        out = out[:, i, :]
        out = out.argmax(dim=1).detach()
        target[:, i + 1] = out
    return target.cpu()


model = Transformer().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)

for epoch in range(1):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        # 在训练时,是拿y的每一个字符输入,预测下一个字符,所以不需要最后一个字
        pred = model(x, y[:, :-1])
        pred = pred.reshape(-1, 39)
        y = y[:, 1:].reshape(-1)
        select = y != zidian_y['<PAD>']
        pred = pred[select]
        y = y[select]
        loss = loss_func(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % 200 == 0:
            pred = pred.argmax(1)
            correct = (pred == y).sum().item()
            accuracy = correct / len(pred)
            lr = optim.param_groups[0]['lr']
            print(epoch, i, lr, loss.item(), accuracy)
    sched.step()

# 测试
for i, (x, y) in enumerate(loader):
    x = x.to(device)
    y = y.to(device)
    break

for i in range(8):
    print(i)
    print(''.join([zidian_xr[i] for i in x[i].tolist()]))
    print(''.join([zidian_yr[i] for i in y[i].tolist()]))
    print(''.join([zidian_yr[i] for i in predict(x[i].unsqueeze(0))[0].tolist()]))
