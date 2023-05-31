import prog.model.model as model 
import prog.model.dataloader as dataloader 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def image_classification_ONE_or_TWO():
    
    net = model.Net()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    nll_loss = nn.NLLLoss()

    num_epochs = 30

    for epoch in range(num_epochs):

     print('Epoch {}/{}'.format(epoch+1, num_epochs))
     print('-------------')

     for phase in ['train', 'valid']:

        if phase == 'train':
            # モデルを訓練モードに設定
            net.train()
        else:
            # モデルを推論モードに設定
            net.eval()

        # 損失和

        epoch_loss = 0.0

        # 正解数

        epoch_corrects = 0

        # DataLoaderからデータをバッチごとに取り出す
        for inputs, labels in dataloader.dataloaders_dict[phase]:
            
            # optimizerの初期化
            optimizer.zero_grad()

            # 学習時のみ勾配を計算させる設定にする
            with torch.set_grad_enabled(phase == 'train'):

                 outputs = net(inputs)

                 # 損失を計算
                 loss = criterion(outputs, labels)
                
                 # ラベルを予測
                 _, preds = torch.max(outputs, 1)
                
                 # 訓練時はバックプロパゲーション
                 if phase == 'train':

                     # 逆伝搬の計算
                     loss.backward()
                     # パラメータの更新
                     optimizer.step()
                
                # イテレーション結果の計算
                # lossの合計を更新
                # PyTorchの仕様上各バッチ内での平均のlossが計算される。
                # データ数を掛けることで平均から合計に変換をしている。
                # 損失和は「全データの損失/データ数」で計算されるため、
                

                # 平均のままだと損失和を求めることができないため。
            epoch_loss += loss.item() * inputs.size(0)
                
                # 正解数の合計を更新
            epoch_corrects += torch.sum(preds == labels.data)

        # epochごとのlossと正解率を表示

        epoch_loss = epoch_loss / len(dataloader.dataloaders_dict[phase].dataset)
        epoch_acc = epoch_corrects.double() / len(dataloader.dataloaders_dict[phase].dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
