import numpy as np

def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    # Poly learning rate policy
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
 
def _fast_hist(label_true, label_pred, n_class):
    # Compute the confusion matrix
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask].astype(int), minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class=19):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / (hist.sum(axis=1) + 1e-12)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-12)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

def fast_hist(a, b, n):
    # Compute the confusion matrix
    # a and b are flattened label arrays
    # n is the number of classes
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    # Compute per-class Intersection over Union
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    # Map labels according to the provided mapping
    # mapping: list of tuples (old_label, new_label)
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def eval_image(predict,label,num_classes):
    # Evaluate predictions against ground truth labels
    index = np.where((label>=0) & (label<num_classes))
    predict = predict[index]
    label = label[index] 
    
    TP = np.zeros((num_classes, 1))
    FP = np.zeros((num_classes, 1))
    TN = np.zeros((num_classes, 1))
    FN = np.zeros((num_classes, 1))
    
    # Calculate TP, FP, TN, FN for each class
    for i in range(0,num_classes):
        TP[i] = np.sum(label[np.where(predict==i)]==i)
        FP[i] = np.sum(label[np.where(predict==i)]!=i)
        TN[i] = np.sum(label[np.where(predict!=i)]!=i)
        FN[i] = np.sum(label[np.where(predict!=i)]==i)        
    
    return TP,FP,TN,FN,len(label)

def create_rgb_composite(patch_data):
    # Create RGB composite from multi-channel patch data
    rgb_patch = patch_data[[3, 2, 1], :, :].transpose(1, 2, 0)
    return rgb_patch

def create_gt_visualization(patch_gt):
    # Create RGB visualization of ground truth labels
    # Map pixel value 1 to red and 0 to white
    gt_visualization = np.zeros(patch_gt.shape + (3,), dtype=np.uint8)
    gt_visualization[patch_gt == 1] = [255, 0, 0]  # Red color for pixel value 1
    gt_visualization[patch_gt == 0] = [255, 255, 255]  # White color for pixel value 0
    return gt_visualization