import numpy as np
import torch
from model import neural_net
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score
from input_pipeline import Dataset


def test(model, db, gpu, logs=None, step=np.nan):

    out = [[] for _ in range(len(db.attr_inds))]
    gt = [[] for _ in range(len(db.attr_inds))]
    with torch.no_grad():
        for test_aux in db.aux_list:
            image_features = torch.from_numpy(db.image_feature[[db.code_list[el] for el in db.input_points_testing[test_aux]], :]).type(torch.FloatTensor)

            if gpu is not None:
                image_features = image_features.cuda(gpu)

            for i in range(len(db.attr_inds)):
                out[i].append(model(image_features)[i].mean(0).item())
                gt[i].append(db.aux_data[test_aux][db.attr_inds[i]])

    for attr_ind in range(len(db.attr_inds)):
        print 'Attribute', db.attr_names[attr_ind]

        r2 = r2_score(gt[attr_ind], out[attr_ind])
        print 'R2 score:', r2

        spr = spearmanr(out[attr_ind], gt[attr_ind]).correlation
        print 'Spearmans corr', spr

        prs = pearsonr(out[attr_ind], gt[attr_ind])[0]
        print 'Pearsons corr ', prs

        if step == step:
            logs['prs'].log_value('correlation_' + db.attr_names[attr_ind], prs, step)
            logs['spr'].log_value('correlation_' + db.attr_names[attr_ind], spr, step)
            logs['r2'].log_value('r2_' + db.attr_names[attr_ind], r2, step)

    return


if __name__ == '__main__':

    half_precision = 0
    gpu = 0 #so far on CPU is not supported

    attr_names = {'Fun_pct': "Fun", "Upper_Class_pct": "Upper Class", "Brand_Asset_C": "Brand Asset"}
    attrs = ['Fun_pct', 'Upper_Class_pct', 'Brand_Asset_C']

    attr_path = 'bav_attributes.txt'
    min_images = 200

    data_path = '../data/brand_dataset/'
    training_path = data_path + 'data/training.csv'
    testing_path = data_path + 'data/testing.csv'
    aux_path = data_path + 'aux_data.csv'
    post_map_path = data_path + 'features/map_list.pickle'
    feature_path = data_path + 'features/features.npy'
    model_path = 'log/2019-06-11_09:09:49.421919/vgg_model_ep_2000.dat'

    db = Dataset(training_path, testing_path, post_map_path, feature_path, aux_path, attr_path, min_images)
    model = neural_net(num_attributes=len(db.attr_inds), aux_size=len(db.aux_list))

    if gpu is not None:
        model.cuda(gpu)
        model.load_state_dict(torch.load(model_path, map_location='cuda:'+ str(gpu)))
    else:
        model.load_state_dict(torch.load(model_path, map_location=lambda gpu, loc: gpu))

    model.eval()

    test(model, db, gpu)
    pass

