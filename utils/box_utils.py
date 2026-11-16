import numpy as np

def point_form(boxes):
    """将中心坐标转换为边界坐标 (xmin, ymin, xmax, ymax)"""
    return np.concatenate((boxes[:, :2] - boxes[:, 2:]/2,
                          boxes[:, :2] + boxes[:, 2:]/2), axis=1)

def center_size(boxes):
    """将边界坐标转换为中心坐标 (cx, cy, w, h)"""
    return np.concatenate(((boxes[:, 2:] + boxes[:, :2])/2,
                          boxes[:, 2:] - boxes[:, :2]), axis=1)

def intersect(box_a, box_b):
    """计算两组边界框的交集面积"""
    A, B = box_a.shape[0], box_b.shape[0]
    
    # 扩展维度以便广播
    max_xy = np.minimum(
        box_a[:, 2:].reshape(A, 1, 2),
        box_b[:, 2:].reshape(1, B, 2)
    )
    min_xy = np.maximum(
        box_a[:, :2].reshape(A, 1, 2),
        box_b[:, :2].reshape(1, B, 2)
    )
    
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=None)
    return inter[..., 0] * inter[..., 1]

def jaccard(box_a, box_b):
    """计算两组边界框的IoU"""
    inter = intersect(box_a, box_b)
    
    # 计算各自面积
    area_a = ((box_a[:, 2]-box_a[:, 0]) * 
             (box_a[:, 3]-box_a[:, 1]))[:, np.newaxis]
    area_b = ((box_b[:, 2]-box_b[:, 0]) * 
             (box_b[:, 3]-box_b[:, 1]))[np.newaxis, :]
    
    union = area_a + area_b - inter
    return inter / np.where(union > 0, union, 1e-10)

def matrix_iou(a, b):
    """批量计算IoU (保留原始实现)"""
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    inter = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, np.newaxis] + area_b - inter + 1e-10)

def matrix_iof(a, b):
    """批量计算IoF"""
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    inter = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return inter / np.maximum(area_a[:, np.newaxis], 1)

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """匹配先验框与真实框"""
    overlaps = jaccard(truths, point_form(priors))
    
    # 找到最佳匹配
    best_prior_overlap = overlaps.max(axis=1, keepdims=True)
    best_prior_idx = overlaps.argmax(axis=1, keepdims=True)

    # 过滤低质量真实框
    valid_mask = best_prior_overlap[:, 0] >= 0.2
    if not np.any(valid_mask):
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # 找到最佳真实框匹配
    best_truth_overlap = overlaps.max(axis=0, keepdims=True)
    best_truth_idx = overlaps.argmax(axis=0)
    
    # 强制匹配最佳先验框
    for j in range(best_prior_idx.shape[0]):
        best_truth_idx[best_prior_idx[j]] = j
    
    # 获取匹配结果
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx]
    conf[best_truth_overlap.flatten() < threshold] = 0
    
    # 编码位置信息
    loc = encode(matches, priors, variances)
    
    # 处理关键点
    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    
    # 存储结果
    loc_t[idx] = loc
    conf_t[idx] = conf
    landm_t[idx] = landm

def encode(matched, priors, variances):
    """编码边界框偏移量"""
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = np.log(g_wh) / variances[1]
    
    return np.concatenate([g_cxcy, g_wh], axis=1)

def encode_landm(matched, priors, variances):
    """编码关键点偏移量"""
    matched = matched.reshape(-1, 5, 2)
    priors = np.broadcast_to(priors[:, None, :], (priors.shape[0], 5, 4))
    
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    
    return g_cxcy.reshape(g_cxcy.shape[0], -1)

def np_decode(loc, priors, variances):
    """解码预测的边界框"""
    centers = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])
    
    return np.concatenate((
        centers - sizes/2,
        centers + sizes/2
    ), axis=1)

def np_decode_landm(pre, priors, variances):
    """解码预测的关键点"""
    return np.concatenate([
        priors[:, :2] + pre[:, i:i+2] * variances[0] * priors[:, 2:]
        for i in range(0, 10, 2)
    ], axis=1)

def nms(boxes, scores, overlap=0.5, top_k=200):
    """非极大值抑制"""
    keep = []
    if boxes.size == 0:
        return np.array(keep)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    # 按分数排序
    order = np.argsort(scores)[-top_k:]
    
    while order.size > 0:
        i = order[-1]
        keep.append(i)
        
        # 计算当前框与其他框的交集
        xx1 = np.maximum(x1[i], x1[order[:-1]])
        yy1 = np.maximum(y1[i], y1[order[:-1]])
        xx2 = np.minimum(x2[i], x2[order[:-1]])
        yy2 = np.minimum(y2[i], y2[order[:-1]])
        
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        
        # 计算IoU
        rem_areas = areas[order[:-1]]
        union = (rem_areas - inter) + areas[i]
        iou = inter / union
        
        # 保留低重叠框
        order = order[:-1][iou <= overlap]
    
    return np.array(keep), len(keep)