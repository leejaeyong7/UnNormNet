def surface_normal_loss(surface_normals, dense_corrs, rotmat):
    '''
    surface_normals: 2x3xHxW surface normal values
    dense_corrs: 2xHxWx2 correspondence representing ref->src coordinates
                 NaN if correspondence is not found / out of range
    rotmat: 1x3x3 rotation matrix from in-plane rotation
    '''

    corrs = dense_corrs.view(2, -1, 2)
    invalid_corrs = (corrs != corrs).any(2).any(0)
    valid_corrs = corrs[:, ~invalid_corrs]
    ref_valid_corrs = valid_corrs[0].long()
    src_valid_corrs = valid_corrs[1].long()

    # first we want to rotate surface normal based on in-plane rotation
    ref_sn = surface_normals[0].permute(1, 2, 0)
    src_sn = surface_normals[1].permute(1, 2, 0)
    warped_ref_sn = ref_sn.matmul(rotmat.transpose(1, 2))
    warped_valid_ref_sn = warped_ref_sn[ref_valid_corrs[:, 0], ref_valid_corrs[:, 1]]
    valid_src_sn = src_sn[src_valid_corrs[:, 0], src_valid_corrs[:, 1]]
    cos_angles = 1 - (warped_valid_ref_sn * valid_src_sn).sum(1)
    loss = cos_angles.mean()

    return loss

