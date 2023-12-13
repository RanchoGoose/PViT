from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc

def to_np(x):
    if isinstance(x, float):
        return np.array([x])
    else:
        return x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin_dual(args, loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    if is_vit_model:   
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // bs and in_dist is False:
                break
                
            data = data.cuda()
            data = Variable(data, requires_grad = True)

            output = net(data)
            if args.prior_model_name == 'resnet18_imagenet200':
                output = output
            else:
                output = output.logits
            smax = to_np(F.softmax(output, dim=1))

            odin_score = (data, output,net, T, noise)
            output_np = to_np(odin_score)
            max_output = np.max(output_np, axis=1)
            _score.append(-max_output)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
    else:
        for batch_idx, (images_32, images_224, target) in enumerate(loader):
            if batch_idx >= ood_num_examples // bs and in_dist is False:
                break
            
            images_32, images_224 = images_32.cuda(), images_224.cuda()
            images_32 = Variable(images_32, requires_grad = True)
            images_224 = Variable(images_224, requires_grad = True)

            priors = net(images_32)
            smax = to_np(F.softmax(priors, dim=1))
            # output = net(images_32)

            odin_score = (images_32, output,net, T, noise)
            output_np = to_np(odin_score)
            max_output = np.max(output_np, axis=1)
            _score.append(-max_output)

            if in_dist:
                preds = np.argmax(smax, axis=1)
                targets = target.numpy().squeeze()
                right_indices = preds == targets
                wrong_indices = np.invert(right_indices)

                _right_score.append(-np.max(smax[right_indices], axis=1))
                _wrong_score.append(-np.max(smax[wrong_indices], axis=1))
                
    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()
    
def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= ood_num_examples // bs and in_dist is False:
            break
        data = data.cuda()
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:
        return concat(_score)[:ood_num_examples].copy()

def ODIN(inputs, outputs, model, temper, noiseMagnitude1):
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    outputs = outputs / temper

    labels = torch.LongTensor(maxIndexTemp).cuda()
    loss = criterion(outputs, labels)
    loss.backward()

    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    gradient[:,0] /= (63.0/255.0)
    gradient[:,1] /= (62.1/255.0)
    gradient[:,2] /= (66.7/255.0)

    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    with torch.no_grad():
        outputs = model(tempInputs)
        outputs = outputs / temper
        nnOutputs = F.softmax(outputs, dim=1)

    return nnOutputs.cpu().numpy()


def get_Mahalanobis_score(model, test_loader, num_classes, sample_mean, precision, layer_index, magnitude, num_batches, in_dist=False):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    
    for batch_idx, batch in enumerate(test_loader):
        if batch_idx >= num_batches and in_dist is False:
            break
        if is_vit_model: 
            data, target = batch
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad = True), Variable(target)
            out_features = model.intermediate_forward(data, layer_index)
        else:
            data_32, data_224, target = batch
            data_32, data_224, target = data_32.cuda(), data_224.cuda(), target.cuda()
            data_32, data_224, target = Variable(data_32, requires_grad = True), Variable(data_224, requires_grad = True), Variable(target)
            out_features = model.intermediate_forward(data_32, layer_index)
        
        # data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, requires_grad = True), Variable(target)
        
        # out_features = model.intermediate_forward(data, layer_index)

        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        
        tempInputs = torch.add(data.data, -magnitude, gradient)
        with torch.no_grad():
            noise_out_features = model.intermediate_forward(tempInputs, layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(-noise_gaussian_score.cpu().numpy())
        
    return np.asarray(Mahalanobis, dtype=np.float32)

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    is_vit_model = any(keyword in args.prior_model_name.lower() for keyword in ["vit", "BEiT"]) or 'imagenet' in args.dataset.lower()
    
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    for batch in train_loader:
        if is_vit_model: 
            data, target = batch
            total += data.size(0)
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad = True), Variable(target)
            output, out_features = model.feature_list(data)
        else:
            data_32, data_224, target = batch
            total += data_32.size(0)
            data_32, data_224, target = data_32.cuda(), data_224.cuda(), target.cuda()
            data_32, data_224, target = Variable(data_32, requires_grad = True), Variable(data_224, requires_grad = True), Variable(target)
            output, out_features = model.feature_list(data_32)
            
        # total += data.size(0)
        # data = data.cuda()
        # data = Variable(data, volatile=True)
        # output, out_features = model.feature_list(data)
        
        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []

    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))
    breakpoint()
    return sample_class_mean, precision
