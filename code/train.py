from model import neural_net
from torch.nn import MSELoss, MarginRankingLoss
import torch.optim as optim
from input_pipeline import Dataset
from torch.utils.data import DataLoader
import datetime
from test import *
from tqdm import tqdm
import os
import argparse
import json
import glob
import tensorboard_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run Training.")
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to use.')
    parser.add_argument('--fixed_std', type=int, default=0,
                        help='fixed or learned std')
    parser.add_argument('--attr_path', nargs='?', default=None,
                    help='file with attribute list')
    parser.add_argument('--resume_train', nargs='?', default=None,
                    help='if to resume train from file')
    parser.add_argument('--dataset_path', nargs='?', default=None,
                    help='location of dataset')
    return parser.parse_args()


def persist_model(model, path):
    torch.save(model.state_dict(), path)
    return


def train(gpu=None):
    logs = {
    'train':tensorboard_logger.Logger(tb_path + "/train"),
    'prs':tensorboard_logger.Logger(tb_path + "/prs"),
    'spr':tensorboard_logger.Logger(tb_path + "/sp"),
    'r2':tensorboard_logger.Logger(tb_path + "/r2"),
    }

    db = Dataset(training_path, testing_path, post_map_path, feature_path, aux_path, attr_path, settings['min_images'])
    print 'Training Attributes:', db.attr_names

    model = neural_net(num_attributes=len(db.attr_inds), aux_size=len(db.aux_list))

    if resume_train is None:
        start_train = 0
    else:
        epochs_str = [el.split('_')[-1].split('.')[0] for el in glob.glob('log/' + resume_train + "/*.dat")]
        if 'model' in epochs_str:
            epochs_str.remove('model')
        last_epoch = np.max([int(el) for el in epochs_str])
        # last_epoch = np.max([int(el.split('_')[-1][0]) for el in glob.glob('log/' + resume_train + "/*.dat")])
        resume_path = 'log/' + resume_train + "/vgg_model_ep_" + str(last_epoch) + ".dat"
        start_train = last_epoch + 1
        if gpu is not None:
            model.load_state_dict(torch.load(resume_path, map_location='cuda:' + str(gpu)))
        else:
            model.load_state_dict(torch.load(resume_path, map_location=lambda gpu, loc: gpu))

    # Initializing PyTorch Dataloader
    dataloader = DataLoader(db, batch_size=settings['batch_size'], shuffle=True, num_workers=4)

    mr_loss = MarginRankingLoss(margin=0.3).to(gpu)

    optimizer = optim.Adadelta(model.parameters(), lr=settings['lr'], weight_decay=1e-5)

    model = model.to(gpu)

    step = 0
    for epoch in range(start_train, settings['num_epochs']):
        print 'Epoch', epoch
        pbar = tqdm(total=db.__len__())
        for i_batch, sample_batched in enumerate(dataloader):
            optimizer.zero_grad()
            image_1 = sample_batched['image_1'].type(torch.FloatTensor)
            image_2 = sample_batched['image_2'].type(torch.FloatTensor)

            aux_1 = sample_batched['label_1'].type(torch.FloatTensor).to(gpu)
            aux_2 = sample_batched['label_2'].type(torch.FloatTensor).to(gpu)

            gt = (aux_1 > aux_2).type(torch.FloatTensor)

            reg_loss_1 = torch.zeros(image_1.shape[0], dtype=torch.float32)
            reg_loss_2 = torch.zeros(image_1.shape[0], dtype=torch.float32)
            ranking_loss = torch.zeros(image_1.shape[0], dtype=torch.float32)

            if gpu is not None:
                image_1 = image_1.to(gpu)
                image_2 = image_2.to(gpu)
                aux_1 = aux_1.to(gpu)
                aux_2 = aux_2.to(gpu)
                gt = gt.to(gpu)
                reg_loss_1 = reg_loss_1.to(gpu)
                reg_loss_2 = reg_loss_2.to(gpu)
                ranking_loss = ranking_loss.to(gpu)

            out_1 = model(image_1)
            out_2 = model(image_2)

            for i in range(len(db.attr_inds)):  # avg over attributes
                ranking_loss += mr_loss(out_1[i], out_2[i], gt[:, i])
            ranking_loss = ranking_loss / len(db.attr_inds)

            if fixed_std:
                p = [torch.distributions.normal.Normal(aux_1[:, i], 0.1) for i in range(len(db.attr_inds))]
                q = [torch.distributions.normal.Normal(out_1[i].mean(1).squeeze(), out_1[i].std(1).squeeze()) for i in range(len(db.attr_inds))]
                for i in range(len(db.attr_inds)):  # avg over attributes
                    reg_loss_1 += torch.distributions.kl.kl_divergence(p[i], q[i])
                reg_loss_1 = reg_loss_1 / len(db.attr_inds)
                p = [torch.distributions.normal.Normal(aux_2[:, i], 0.1) for i in range(len(db.attr_inds))]
                q = [torch.distributions.normal.Normal(out_2[i].mean(1).squeeze(), out_2[i].std(1).squeeze()) for i in range(len(db.attr_inds))]
                for i in range(len(db.attr_inds)):  # avg over attributes
                    reg_loss_2 += torch.distributions.kl.kl_divergence(p[i], q[i])
                reg_loss_2 = reg_loss_2 / len(db.attr_inds)
            else:
                p = [torch.distributions.normal.Normal(aux_1[:, i], model.aux_stds[sample_batched['aux_1'], i]) for i in range(len(db.attr_inds))]
                q = [torch.distributions.normal.Normal(out_1[i].mean(1).squeeze(), out_1[i].std(1).squeeze()) for i in range(len(db.attr_inds))]
                for i in range(len(db.attr_inds)):  # avg over attributes
                    reg_loss_1 += torch.distributions.kl.kl_divergence(p[i], q[i])
                reg_loss_1 = reg_loss_1 / len(db.attr_inds)
                p = [torch.distributions.normal.Normal(aux_2[:, i], model.aux_stds[sample_batched['aux_2'], i]) for i in range(len(db.attr_inds))]
                q = [torch.distributions.normal.Normal(out_2[i].mean(1).squeeze(), out_2[i].std(1).squeeze()) for i in range(len(db.attr_inds))]
                for i in range(len(db.attr_inds)):  # avg over attributes
                    reg_loss_2 += torch.distributions.kl.kl_divergence(p[i], q[i])
                reg_loss_2 = reg_loss_2 / len(db.attr_inds)

            ranking_loss = ranking_loss.mean()  # avg over batch
            reg_loss = reg_loss_1.mean() + reg_loss_2.mean()  # avg over batch

            loss = reg_loss + ranking_loss

            step += 1
            logs['train'].log_value('loss', loss.item(), step)

            loss.backward()
            optimizer.step()

            _loss = loss.item()
            pbar.update(image_1.shape[0])

        pbar.close()

        if epoch % 50 == 0:
            model.eval()
            test(model, db, gpu, logs=logs, step=step)
            model.train()
            persist_model(model, experiment_folder + '/vgg_model_ep_' + str(epoch) + '.dat')

    # Performing final evaluation
    model.eval()
    test(model, db, gpu)
    persist_model(model, model_path)
    return


if __name__ == '__main__':

    args = parse_args()

    gpu = args.gpu
    resume_train = args.resume_train
    attr_path = args.attr_path
    fixed_std = args.fixed_std
    data_path = args.dataset_path

    training_path = data_path + 'data/training.csv'
    testing_path = data_path + 'data/testing.csv'
    aux_path = data_path + 'aux_data.csv'
    post_map_path = data_path + 'features/map_list.pickle'
    feature_path = data_path + 'features/features.npy'

    with open(data_path + 'settings.json', 'r') as f:
        settings = json.load(f)

    experiment_time = str(datetime.datetime.now()).replace(' ', '_')
    if resume_train is None:
        experiment_folder = 'log/' + experiment_time
    else:
        experiment_folder = 'log/' + resume_train

    model_path = experiment_folder + '/vgg_model.dat'
    tb_path = experiment_folder + '/tb'

    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder)
        print 'Experiment log folder is: ' + experiment_folder
        os.makedirs(experiment_folder + '/losses')
    if not resume_train:
        os.makedirs(tb_path)


    train(gpu)

