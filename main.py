import torch
from torch import nn
from torchvision import transforms
import os
import dlib
from argparse import Namespace

# files from e4e
from models.psp import pSp
from models.FTM import FaceTransferModule
import torch_utils
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
# our own py files
from Dataloader import get_dataloader
import our_utils
from criteria.id_loss import IDLoss
# 导入criteria.id_loss模块中的IDLoss类，这是一个定义身份识别损失的类，用于保持生成图像的身份特征。

from criteria.face_parsing.face_parsing_loss import FaceParsingLoss
# 导入criteria.face_parsing.face_parsing_loss模块中的FaceParsingLoss类，这是一个面部解析损失类，可能用于评估面部区域的准确性。

from criteria.lpips.lpips import LPIPS
# 导入criteria.lpips.lpips模块中的LPIPS类，这是一个感知图像补丁相似性指标，常用于评估图像质量。

from criteria.style_loss import StyleLoss

from criteria.norm import NormLoss
from configs.paths_config import ModelPath

def parse_images(img, target, recon1, display_count=2):
    im_data = []
    display_count = min(display_count, len(img))
    for i in range(display_count):
        cur_im_data = {
            'input_face': torch_utils.tensor2im(img[i]),
            'target_face': torch_utils.tensor2im(target[i]),
            'recon_styleCode': torch_utils.tensor2im(recon1[i]),
        }
        im_data.append(cur_im_data)
    # 将一批图像数据转换为可视化的格式，用于记录或显示
    return im_data

def log_images(name, imgs1_data, epoch=0, subscript=None, log_latest=False):
    fig = torch_utils.vis_faces(imgs1_data)
    step = epoch if not log_latest else 0
    if subscript:
        path = os.path.join(args.save_path, name, f'{subscript}_{step:06d}.jpg')
    else:
        path = os.path.join(args.save_path, name, f'{step:06d}.jpg')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)

def log_metrics(metrics_dict, prefix, epoch, logger):
    for key, value in metrics_dict.items():
        logger.add_scalar(f'{prefix}/{key}', value, epoch)
    # 向TensorBoard记录标量数据

def print_metrics(metrics_dict, prefix, epoch):
    print(f'Metrics for {prefix}, epoch {epoch}')
    for key, value in metrics_dict.items():
        print(f'\t{key} = ', value)
    # 打印训练或测试时的损失和性能指标

def load_from_train_checkpoint(self, ckpt):
    print('Loading previous training data...')
    self.global_step = ckpt['global_step'] + 1
    self.best_val_loss = ckpt['best_val_loss']
    self.net.load_state_dict(ckpt['state_dict'])

    if self.opts.keep_optimizer:
        self.optimizer.load_state_dict(ckpt['optimizer'])
    if self.opts.w_discriminator_lambda > 0:
        self.discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
    if self.opts.progressive_steps:
        self.check_for_progressive_training_update(is_resume_from_ckpt=True)
    print(f'Resuming training from step {self.global_step}')



def checkpoint_me(checkpoint_dir, save_dict, is_best):
    save_name = 'best_model.pt' if is_best else f'iteration_{save_dict["epoch"]}.pt'
    # 根据是否是最佳模型来决定保存的文件名

    #save_dict = __get_save_dict(model)
    # 获取当前模型和优化器的状态字典

    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    # 设置检查点的保存路径

    torch.save(save_dict, checkpoint_path)
    # 保存检查点到文件

    with open(os.path.join(checkpoint_dir, 'timestamp.txt'), 'a') as f:
        if is_best:
            f.write(f'**Best**: Epoch - {save_dict["epoch"]}, Loss - {save_dict["best_val_loss"]} \n{save_dict["loss_dict"]}\n')
        else:
            f.write(f'Step - {save_dict["epoch"]}, \n{save_dict["loss_dict"]}\n')

def __get_save_dict(model, epoch, loss_dict, best_loss):
    save_dict = {
        'state_dict': model.state_dict(),
        'epoch': epoch,
        'loss_dict': loss_dict,
        'best_val_loss': best_loss,
        'opts': vars(args)
    }
    # 保存模型的状态、配置选项
    return save_dict
    # 返回包含所有保存信息的字典


def calc_loss(img, recon1, latent_source, out_latent):
    loss_dict = {}
    loss = 0.0
    id_logs = None
    # 这部分初始化了损失字典loss_dict用于存储各种损失的具体值，loss用于累计总损失，id_logs用于可能的身份验证损失的附加信息。

    if args.face_parsing_lambda > 0:
        face_parsing_loss = FaceParsingLoss(args).cuda().eval()
        loss_face_parsing_1, face_parsing_sim_improvement_1 = face_parsing_loss(recon1, img)

        loss_dict['loss_face_parsing'] = float(loss_face_parsing_1)
        loss_dict['face_parsing_improve'] = float(face_parsing_sim_improvement_1)
        loss += loss_face_parsing_1 * args.face_parsing_lambda
    # 如果面部解析权重系数face_parsing_lambda大于0，计算面部解析损失并加权累加到总损失中。这有助于模型在重建过程中更好地捕捉面部的特征。

    if args.id_lambda > 0:
        id_loss = IDLoss(args).cuda().eval()
        loss_id_1, sim_improvement_1, id_logs_1 = id_loss(recon1, img)

        loss_dict['loss_id'] = float(loss_id_1)
        loss_dict['id_improve'] = float(sim_improvement_1)
        loss += loss_id_1 * args.id_lambda
    # 同样，如果身份识别损失的权重系数id_lambda大于0，计算并记录身份识别损失，并根据配置的权重累加到总损失中。
    if args.l2_lambda > 0:
        mse_loss = nn.MSELoss().cuda().eval()
        #print(f'recon1 size: {recon1.shape}')
        #print(f'img size: {img.shape}')
        loss_l2_1 = F.mse_loss(recon1, img)

        loss_dict['loss_l2'] = float(loss_l2_1)
        loss += loss_l2_1 * args.l2_lambda
    # L2损失（均方误差损失）用于评估重建图像与原图像间的像素级误差，如果设置了其权重，则加入总损失中。
    if args.lpips_lambda > 0:
        lpips_loss = LPIPS(net_type='alex').cuda().eval()
        loss_lpips = 0
        for i in range(3):
            loss_lpips_1 = lpips_loss(
                F.adaptive_avg_pool2d(recon1, (256 // 2 ** i, 256 // 2 ** i)),
                F.adaptive_avg_pool2d(img, (256 // 2 ** i, 256 // 2 ** i))
            )

            loss_lpips += loss_lpips_1

        loss_dict['loss_lpips'] = float(loss_lpips)
        loss += loss_lpips * args.lpips_lambda

    if args.norm_lambda > 0:
        norm_loss = NormLoss()
        loss_norm = norm_loss(latent_source, out_latent)
        loss_dict['loss_norm'] = float(loss_norm)
        loss += loss_norm * args.norm_lambda

    # LPIPS损失评估重建图像与原图像在感知相似性上的差异，它用于捕捉人类视觉系统可能察觉到的差异，而不仅仅是像素级的差异。
    """
    if self.opts.w_norm_lambda > 0:
        loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)

        loss_dict['loss_w_norm'] = float(loss_w_norm)
        loss += loss_w_norm * self.opts.w_norm_lambda
    #W-Norm损失用于保持潜在空间的分布稳定，有助于生成的一致性和质量。
    """
    """
    if self.opts.style_lambda > 0:  # gram matrix loss
        loss_style_1 = self.style_loss(recon1, img, mask_x=(mask == 3).float(), mask_x_hat=(mask == 3).float())

        loss_dict['loss_style'] = float(loss_style_1)
        loss += loss_style_1 * self.opts.style_lambda

    """
    # 风格损失（Gram matrix loss）评估图像在风格上的相似性，常用于艺术风格迁移等应用。
    # 最后，总损失被加总并记录，函数返回包括总损失、损失字典和可能的身份日志的元组。
    loss_dict['loss'] = float(loss)
    print(f'loss: {loss_dict}')
    return (loss, loss_dict, id_logs_1) if args.id_lambda > 0 else (loss, loss_dict, None)

def main(args):
    torch.cuda.empty_cache()
    stylegan_ffhq_path = os.path.join(args.pretrained_path, 'stylegan2-ffhq-config-f.pt')
    ir_se50_path = os.path.join(args.pretrained_path, 'model_ir_se50.pth')
    shape_predictor_path = os.path.join(args.pretrained_path, 'shape_predictor_68_face_landmarks.dat')
    moco_path = os.path.join(args.pretrained_path, 'moco_v2_800ep_pretrain.pth')
    face_parsing_path = os.path.join(args.pretrained_path, 'face_parsing_model.pth')
    model_paths = {
        'stylegan_ffhq': stylegan_ffhq_path,
        'ir_se50': ir_se50_path,
        'shape_predictor': shape_predictor_path,
        'moco': moco_path,
        'face_parsing': face_parsing_path
    }
    # Initialize tensorboard logger
    log_dir = os.path.join(args.save_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(logdir=log_dir)
    # 初始化Tensorboard日志记录器，只有rank为0的进程会写日志。

    # Initialize checkpoint dir
    checkpoint_dir = os.path.join(args.save_path, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 初始化检查点保存目录，以及相关的保存间隔和验证损失记录。
    resize_dims = (256,256)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    print("loading predictor for face align(denied for now)")
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    # fix random seeds
   # torch.manual_seed(args.seed)
   # np.random.seed(args.seed)
   # torch.use_deterministic_algorithms(True)
    print("Getting Data Loader")
    train_loader, val_loader, test_loader = get_dataloader(args.dataset_dir, transform, transform, predictor, args.batch_size)
    #model_path = ModelPath(args)
    #model_path_dict = model_path.get_model_path()
    print('Loading E4E!')
    outside_model_path = os.path.join(args.pretrained_path, 'e4e_ffhq_encode.pt')
    outside_ckpt = torch.load(outside_model_path, map_location='cpu')
    outside_opts = outside_ckpt['opts']
    # pprint.pprint(opts)  # Display full options used
    # update the training options
    outside_opts['checkpoint_path'] = outside_model_path
    outside_opts['model_paths'] = model_paths
    args.model_paths = model_paths
    outside_opts = Namespace(**outside_opts)
    outside_net = pSp(outside_opts)
    outside_net.eval()
    outside_net.cuda()
    # print(net)
    print('E4E Model successfully loaded!')

    # ==== Initialize network ====
    print('Loading FTM!')
    model = FaceTransferModule(feature_size=512, num_blocks=args.batch_size)
    model.cuda()
    print('FTM successfully loaded!')
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)
    # optimizer = torch.optim(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    stat_training_loss = []
    stat_val_loss = []
    best_val_loss = float('inf')
    best_train_loss = None
    stat_training_acc = []
    stat_val_acc = []
    agg_train_loss_dict = []
    agg_valid_loss_dict = []
    save_dict = None
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}=========================================')
        # 虽然每个epoch都 validate一下，但是每间隔3个epoch再显示图片，防止图片太多了
        if args.epochs % args.show_interval == 0:
            show_images = True
        else:
            show_images = False
        training_loss = 0
        training_samples = 0
        val_loss = 0
        val_samples = 0
        # training
        print("Start training!")
        model.train()
        for batch_idx, (train_src_img, train_target_img) in enumerate(train_loader):
            train_src_img = train_src_img.to("cuda")
            train_target_img = train_target_img.to("cuda")
            batch_size = args.batch_size
            optimizer.zero_grad()
            source_latents = our_utils.face_to_latents(train_src_img, outside_net)
            target_latents = our_utils.face_to_latents(train_target_img, outside_net)
            #print(f'source_latents: {source_latents.shape}')
            #print(f'target_latents: {target_latents.shape}')
            manipulated_features = model.forward(source_latents, target_latents)
            #print(f'manipulated_features: {manipulated_features}')
            out_latents = torch.stack(manipulated_features, dim=0)
            #print(f'out_latents: {out_latents.shape}')
            recon_image = our_utils.latents_to_face(out_latents, outside_net)
            recon_image = our_utils.resize_tensor(recon_image, 256, 256)
            #print(f'Resized shape: {recon_image.shape}')
            loss_, loss_dict, id_logs = calc_loss(train_target_img, recon_image, source_latents, out_latents)
            loss_.backward()
            optimizer.step()
           # _, top_class = logits.topk(1, dim=1)
            #training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            #training_loss += batch_size * loss_
            #training_samples += batch_size
            agg_train_loss_dict.append(loss_dict)

        train_loss_dict = torch_utils.aggregate_loss_dict(agg_train_loss_dict)
        log_metrics(train_loss_dict, 'train', epoch, logger)
        print_metrics(train_loss_dict, 'train', epoch)
        # validation
        print("Start validation!")
        model.eval()
        iterations = 0
        for batch_idx, (val_src_img, val_target_img) in enumerate(val_loader):
            iterations += 1
            with torch.no_grad():
                batch_size = args.batch_size
                val_src_img = val_src_img.to("cuda")
                val_target_img = val_target_img.to("cuda")
                source_latents = our_utils.face_to_latents(val_src_img, outside_net)
                target_latents = our_utils.face_to_latents(val_target_img, outside_net)
                manipulated_features = model.forward(source_latents, target_latents)
                out_latents = torch.stack(manipulated_features, dim=0)
                recon_image = our_utils.latents_to_face(out_latents, outside_net)
                recon_image = our_utils.resize_tensor(recon_image, 256, 256)
                loss_, loss_dict, id_logs = calc_loss(val_target_img, recon_image, source_latents, out_latents)
                #equals = top_class == val_labels.cuda().view(*top_class.shape)
                #val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                #val_loss += batch_size * loss_
                #val_samples += batch_size
                agg_valid_loss_dict.append(loss_dict)
                if show_images and iterations % 10 == 0:
                    imgs = parse_images(val_src_img, val_target_img, recon_image)
                    log_images('images/valid/faces', imgs1_data=imgs, epoch=epoch, subscript='{:04d}'.format(batch_idx))
        valid_loss_dict = torch_utils.aggregate_loss_dict(agg_valid_loss_dict)
        log_metrics(valid_loss_dict, 'valid', epoch, logger)
        print_metrics(valid_loss_dict, 'valid', epoch)
        if len(valid_loss_dict) > 0 and valid_loss_dict['loss'] < best_val_loss:
            best_val_loss = valid_loss_dict['loss']
            save_dict = __get_save_dict(model, epoch, valid_loss_dict, best_val_loss)
            if epoch % 1 == 0 or epoch + 1 == args.epochs:
                print("Store Checkpoint!")
                checkpoint_me(checkpoint_dir, save_dict, is_best=True)
        #assert val_samples == 10000
        # update stats
        #stat_training_loss.append(training_loss/training_samples)
        #stat_val_loss.append(val_loss/val_samples)
        #stat_training_acc.append(training_acc/training_samples)
        #stat_val_acc.append(val_acc/val_samples)
        # print
        #print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}")
        # lr scheduler
        scheduler.step()
    #print("Store Checkpoint!")
    #checkpoint_me(checkpoint_dir, save_dict, is_best=True)
    # plot
    #plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
    # test

    if args.test:
        with torch.no_grad():
            for batch_idx, (test_src_img, test_target_img) in enumerate(test_loader):
                model = FaceTransferModule(feature_size=512, num_blocks=args.batch_size)
                checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
                ckpt = torch.load(outside_model_path, map_location='cpu')
                opts = ckpt['opts']
                # print.pprint(opts)  # Display full options used
                # update the training options
                opts['save_path'] = checkpoint_path
                opts = Namespace(**outside_opts)
                #outside_net = pSp(outside_opts)
                model.load_state_dict(ckpt['state_dict'])
                model.eval()
                model.cuda()
                source_latents = our_utils.face_to_latents(test_src_img, outside_net)
                target_latents = our_utils.face_to_latents(test_target_img, outside_net)
                manipulated_features = model.forward(source_latents, target_latents)
                out_latents = torch.stack(manipulated_features, dim=0)
                recon_image = our_utils.latents_to_face(out_latents, outside_net)
                imgs = parse_images(test_src_img, test_target_img, recon_image)
                log_images('images/test/faces', imgs1_data=imgs, subscript='{:04d}'.format(batch_idx), log_latest=True)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--pretrained_path', type=str, help='')
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--show_interval', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd', type=float, help='')
    parser.add_argument('--eps', type=float, help='')
    parser.add_argument('--eta_min', type=float, help='')
    #parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    #parser.add_argument('--mixup', action='store_true')
    #parser.set_defaults(mixup=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    #parser.add_argument('--save_images', action='store_true')
    #parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--face_parsing_lambda', type=float, help='')
    parser.add_argument('--id_lambda', type=float, help='')
    parser.add_argument('--id_loss_multiscale', action='store_true')
    parser.set_defaults(id_loss_multiscale=False)
    parser.add_argument('--l2_lambda', type=float, help='')
    parser.add_argument('--lpips_lambda', type=float, help='')
    parser.add_argument('--norm_lambda', type=float, help='')
    args = parser.parse_args()
    print(args)
    main(args)
