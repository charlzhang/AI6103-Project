import torch
import torch.nn as nn
import torch.optim as optim


class TransferCell(nn.Module):
    def __init__(self, feature_size):
        super(TransferCell, self).__init__()
        self.K1 = nn.Linear(feature_size*2, feature_size)
        self.K2 = nn.Linear(feature_size*2, feature_size)

    def forward(self, source_features, target_features):
        # Concatenate source and target features
        concatenated_features = torch.cat((source_features, target_features), dim=1)
        # First non-linear transformation step
        step1 = torch.sigmoid(self.K1(concatenated_features)) * source_features
        # Second non-linear transformation step
        step2 = torch.tanh(self.K2(concatenated_features)) + step1

        return step2

class FaceTransferBlock(nn.Module):
    def __init__(self, feature_size):
        super(FaceTransferBlock, self).__init__()
        # Each Face Transfer Block contains three identical Transfer Cells
        self.transfer_cell1 = TransferCell(feature_size)
        self.transfer_cell2 = TransferCell(feature_size)
        self.transfer_cell3 = TransferCell(feature_size)
        self.weight = nn.Parameter(torch.rand(feature_size))

    def forward(self, source_features, target_features):
        # Process through three Transfer Cells
        processed_features = self.transfer_cell1(source_features, target_features)
        processed_features = self.transfer_cell2(processed_features, target_features)
        processed_features = self.transfer_cell3(processed_features, target_features)
        
        # Weighted sum of processed features
        weighted_sum = torch.sigmoid(self.weight) * processed_features + (1 - torch.sigmoid(self.weight)) * source_features
        return weighted_sum

class FaceTransferModule(nn.Module):
    def __init__(self, feature_size, num_blocks):
        super(FaceTransferModule, self).__init__()
        self.blocks = nn.ModuleList([FaceTransferBlock(feature_size) for _ in range(num_blocks)])

    def forward(self, source_features, target_features):
        manipulated_features = []
        min_length = min(len(source_features), len(target_features), len(self.blocks))
        for i in range(min_length):
            manipulated_feature = self.blocks[i](source_features[i], target_features[i])
            manipulated_features.append(manipulated_feature)
        return manipulated_features
def main():
    batch = 6
    num_epochs = 100
    print_every = 10
    ftm_model = FaceTransferModule(feature_size=512, num_blocks=batch)
    source_latents = torch.rand(batch, 18, 512)
    target_latents = torch.rand(batch, 18, 512)
    # source_latents = our_utils.face_to_latents(source_image,outside_net)
    # target_latents = our_utils.face_to_latents(target_image,outside_net)
    # out_latents = torch.cat((source_latents[:,9:,:],target_latents[:,:9,:]),dim=1)
    # out_latents = prepare_for_stylegan2(manipulated_features)
    optimizer = optim.SGD(ftm_model.parameters(), lr=0.01)  # 使用随机梯度下降优化器

    # 定义损失函数（示例）
    criterion = nn.MSELoss()

    # 进行训练
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # 梯度归零
        # 前向传播
        manipulated_features = ftm_model(source_latents, target_latents)
        out_latents = torch.stack(manipulated_features, dim=0)
        print(out_latents.shape)
        # 计算损失
        loss = criterion(out_latents, target)  # 这里的 target 是你的目标值
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 打印训练信息
    #    if (epoch + 1) % print_every == 0:
    #        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item())
    # for latent in out_latents:
    #     print(latent.shape)
    # images = our_utils.latents_to_face(out_latents,outside_net)
    # example = our_utils.display_alongside_image(tensor2im(source_image[0]),tensor2im(target_image[0]),tensor2im(images[0]))
    # example.save("example.png")
    
    
    return

if __name__ == '__main__':
    main()