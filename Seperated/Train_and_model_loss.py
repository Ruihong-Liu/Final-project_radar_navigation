# loss function
def combined_loss(pred_betas, pred_pose_body, pred_root_orient, pred_trans, pred_vertices, 
                  gt_betas, gt_pose_body, gt_root_orient, gt_trans, gt_vertices, 
                  pred_genders, gt_genders, criterion, gender_criterion, model):
    
    betas_loss = criterion(pred_betas, gt_betas)
    pose_body_loss = criterion(pred_pose_body, gt_pose_body)
    root_orient_loss = criterion(pred_root_orient, gt_root_orient)
    trans_loss = criterion(pred_trans, gt_trans)
    vertices_loss = criterion(pred_vertices, gt_vertices)
    gender_loss = gender_criterion(pred_genders, gt_genders.unsqueeze(1).float())
    
    losses = {
        'betas': betas_loss,
        'pose_body': pose_body_loss,
        'root_orient': root_orient_loss,
        'trans': trans_loss,
        'vertices': vertices_loss,
        'gender': gender_loss
    }

    total_loss = sum(model.loss_weights[key] * loss for key, loss in losses.items())
    
    return total_loss, losses