
def description_loss(desc, corrs):
    '''
    Performs dense description loss
    Args:
        desc(torch.Tensor): N x C x H8 x W8 dense descriptor
        corrs(torch.Tensor): N x H x W dense correspondence
    Returns:
        (float, float): positive, negative loss value for description
    '''
    N, C, Hc, Wc = desc.shape
    N, H, W, _ = corrs.shape
    Hs = H // Hc # 8
    Ws = W // Wc # 8
    assert N == 2

    # take the center value of 8x8 grid at corrs
    # leave out outer edges 
    center_corrs = corrs.view(N, Hc, Hs, Wc, Ws, 2)[:, 3:-3, Hs // 2, 3:-3, Ws // 2]

    # Hc x Wc x 2
    ref_coords = center_corrs[0].contiguous().view(1, -1, 2) # before warp
    src_coords = center_corrs[1].contiguous().view(-1, 1, 2) # after warp
    coord_diff = (ref_coords - src_coords).norm(2, dim=-1)
    M = coord_diff.shape[0]

    # proxy corrs = HcWc x HcWc, where each pixel in row corresponds to
    # source image coordinate.
    # HcWc(ref) x HcWc(src)
    # this is because warped coordinates are ref warped onto src image
    proxy_corrs = (coord_diff <= 7.5).float()
    warped_coords = src_coords.view(1, -1, 2)
    valid_mask = (warped_coords == warped_coords).prod(2).float()
    
    # N x C x HcWc
    dense_desc= desc[:, :, 3:-3, 3:-3].contiguous().view(N, C, -1)
    # normalize dense desc
    norm_dense_desc = dense_desc# / dense_desc.norm(2, dim=1).unsqueeze(1)

    # HcWc (ref) x HcWc (src) distances
    dists = (norm_dense_desc[0].view(C, -1, 1) * norm_dense_desc[1].view(C, 1, -1)).sum(0)
    num_samples = dists.shape[0]

    # positive loss
    lambda_d= 1#250
    positive_margin = 1
    negative_margin = 0.2

    # dists = HcWc x HcWc
    # HcWc x HcWc
    positive_dists = (positive_margin - dists).clamp_min(0)
    # HcWc x HcWc
    negative_dists = (dists - negative_margin).clamp_min(0)
    normalization = (valid_mask.sum() * (Hc * Wc)).float()

    positive_loss = (lambda_d * proxy_corrs * valid_mask * positive_dists).sum() / normalization
    negative_loss = ((1 - proxy_corrs) * valid_mask *  negative_dists).sum() / normalization
    loss = lambda_d * proxy_corrs * positive_dists + (1 - proxy_corrs) * negative_dists
    final_loss = (loss * valid_mask).sum() / normalization


    return final_loss , positive_loss, negative_loss, normalization
