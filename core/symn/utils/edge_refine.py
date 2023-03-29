from os.path import join
import numpy as np
import cv2
import torch
import mmcv


def edge_refine(R: np.ndarray, t: np.ndarray, K, obj_id,
                amodal_mask_prob, visib_mask_prob,
                renderer, vis_dir='', debug=False):
    """
    Optimize estimated pose (R, t) by edge align
    :param R: np.ndarray (3, 3)
    :param t: np.ndarray (3, )
    :param K: np.ndarray (3, 3)
    :param obj_id:
    :param amodal_mask_prob: torch.Tensor (128, 128)
    :param visib_mask_prob: torch.Tensor (128, 128)
    :param renderer:
    :param vis_dir:
    :return: opti_R, opti_t
    """
    if debug:
        mmcv.mkdir_or_exist(vis_dir)

    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if len(t.shape) == 1:  # convert [3,] to  [3, 1]
        t = t[:, np.newaxis]
    #### binarize mask_prob
    if isinstance(amodal_mask_prob, torch.Tensor):
        amodal_mask_prob = amodal_mask_prob.detach().cpu().numpy()
        while len(amodal_mask_prob.shape) > 2:
            amodal_mask_prob = amodal_mask_prob[0]
    amodal_mask = np.zeros(amodal_mask_prob.shape, dtype=np.uint8)
    amodal_mask[amodal_mask_prob > 0.5] = 1
    if isinstance(visib_mask_prob, torch.Tensor):
        visib_mask_prob = visib_mask_prob.detach().cpu().numpy()
        while len(visib_mask_prob.shape) > 2:
            visib_mask_prob = visib_mask_prob[0]
    visib_mask = np.zeros(visib_mask_prob.shape, dtype=np.uint8)
    visib_mask[visib_mask_prob > 0.5] = 1
    #### find visible contours
    contours, _ = cv2.findContours(amodal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    visible_contours = []  # [M, 2]
    for contour in contours:
        for i in range(len(contour)):
            x, y = contour[i, 0]  # [x,y]
            if np.any(visib_mask[y - 1:y + 1, x - 1:x + 1]) and x > 0 and y > 0 and x < 128 and y < 128:
                visible_contours.append(contour[i])
    try:
        visible_contours = np.concatenate(visible_contours)
    except:
        print("some problem, can not find visible_contours")
        return R, t, False
    last_cost = 1e5
    # tikhonov_matrix_ = np.diag([500., 500., 500., ])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for it in range(10):
        if debug:
            contour_image = np.zeros((128, 128, 3), dtype=np.uint8)
        # render image in current R, t
        render = renderer.render(obj_id, K, R, t)
        render_mask = (render[..., 3] == 1.).astype(np.uint8)
        render_contours, _ = cv2.findContours(render_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            render_contours = max(render_contours, key=len)[:, 0, :]  # [N,2]
        except:
            return R, t, False
        depth_img = renderer.render(obj_id, K, R, t, read_depth=True)

        cost = 0
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        count = 0
        for vc in visible_contours:
            distances = np.linalg.norm(vc[np.newaxis, :] - render_contours, axis=1)
            min_index = np.argmin(distances)
            cost += distances[min_index]
            rc = render_contours[min_index]
            e = (vc - rc)[:, np.newaxis]
            if debug:
                cv2.line(contour_image, vc, rc, (128, 128, 128), 1)
            depth = depth_img[rc[1], rc[0]]
            p_camera = np.array([depth * (rc[1] - cx) / fx, depth * (rc[0] - cy) / fy, depth])
            inv_z = 1.0 / p_camera[2]
            inv_z2 = inv_z * inv_z

            x, y = p_camera[0], p_camera[1]
            de_dtheta = np.array([[fx * x * y * inv_z2, - fx - fx * x * x * inv_z2, fx * y * inv_z,
                                   -fx * inv_z, 0, fx * x * inv_z2],
                                  [fy + fy * y * y * inv_z2, -fy * x * y * inv_z2, -fy * x * inv_z,
                                   0, -fy * inv_z, fy * y * inv_z2]])
            # J = de_dtheta, H = J ^ T * J, b = -J ^ T * e
            H += np.matmul(de_dtheta.transpose(), de_dtheta)
            b -= np.matmul(de_dtheta.transpose(), e)
        if debug:
            for vc in visible_contours:
                contour_image[vc[1], vc[0]] = (255, 0, 0)  # blue
            for rc in render_contours:
                contour_image[rc[1], rc[0]] = (0, 0, 255)  # red
        cost = cost / len(visible_contours)
        if debug:
            print(f"iter {it}")
            print(f"cost: {cost}, last cost: {last_cost}")
            if vis_dir == "":
                cv2.namedWindow(f"edge_refine" + str(it) + ".png", cv2.WINDOW_NORMAL)
                cv2.imshow(f"edge_refine" + str(it) + ".png", contour_image)
            else:
                cv2.imwrite(join(vis_dir, "debug" + str(it) + ".png"), contour_image)
        if cost < last_cost:
            last_cost = cost
        else:
            break
        #  Optimize and update pose
        theta = np.linalg.solve(H, b)
        dR, _ = cv2.Rodrigues(theta[0:3])
        dt = theta[3:6]
        if debug:
            print(f"theta, {theta}")
            print(f"dR, {dR}")
            print(f"dt, {dt}")
        R = np.matmul(dR, R)
        t = np.matmul(dR, t) + dt

    return R, t, True

def edge_refine_v2(R: np.ndarray, t: np.ndarray, K, obj_id,
                   amodal_mask_prob, visib_mask_prob,
                   renderer, vis_dir=''):
    """
    Optimize estimated pose (R, t) by edge align


    :param R: np.ndarray (3, 3)
    :param t: np.ndarray (3, )
    :param K: np.ndarray (3, 3)
    :param obj_id:
    :param amodal_mask_prob: torch.Tensor (128, 128)
    :param visib_mask_prob: torch.Tensor (128, 128)
    :param renderer:
    :param vis_dir:
    :return: opti_R, opti_t
    """

    mmcv.mkdir_or_exist(vis_dir)

    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if len(t.shape) == 1:  # convert [3,] to  [3, 1]
        t = t[:, np.newaxis]
    #### binarize mask_prob
    if isinstance(amodal_mask_prob, torch.Tensor):
        amodal_mask_prob = amodal_mask_prob.detach().cpu().numpy()
        while len(amodal_mask_prob.shape) > 2:
            amodal_mask_prob = amodal_mask_prob[0]
    amodal_mask = np.zeros(amodal_mask_prob.shape, dtype=np.uint8)
    amodal_mask[amodal_mask_prob > 0.5] = 1
    if isinstance(visib_mask_prob, torch.Tensor):
        visib_mask_prob = visib_mask_prob.detach().cpu().numpy()
        while len(visib_mask_prob.shape) > 2:
            visib_mask_prob = visib_mask_prob[0]
    visib_mask = np.zeros(visib_mask_prob.shape, dtype=np.uint8)
    visib_mask[visib_mask_prob > 0.5] = 1
    #### find visible contours
    contours, _ = cv2.findContours(amodal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    visible_contours = []  # [M, 2]
    for contour in contours:
        for i in range(len(contour)):
            x, y = contour[i, 0]  # [x,y]
            if np.any(visib_mask[y - 1:y + 1, x - 1:x + 1]) and x > 0 and y > 0 and x < 128 and y < 128:
                visible_contours.append(contour[i])
    try:
        visible_contours = np.concatenate(visible_contours)
    except:
        print("some problem, can not find visible_contours")
        return R, t, False
    last_cost = 1e5
    # tikhonov_matrix_ = np.diag([500., 500., 500., ])
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    for it in range(100):
        contour_image = np.zeros((128, 128, 3), dtype=np.uint8)
        # render image in current R, t
        render = renderer.render(obj_id, K, R, t)
        render_mask = (render[..., 3] == 1.).astype(np.uint8)
        render_contours, _ = cv2.findContours(render_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            render_contours = max(render_contours, key=len)[:, 0, :]  # [N,2]
        except:
            return R, t, False
        coord_img = renderer.denormalize(render[..., :3], obj_id)
        depth_img = renderer.render(obj_id, K, R, t, read_depth=True)

        cost = 0
        H = np.zeros((6, 6))
        b = np.zeros((6, 1))
        count = 0
        for vc in visible_contours:
            distances = np.linalg.norm(vc[np.newaxis, :] - render_contours, axis=1)
            min_index = np.argmin(distances)
            cost += distances[min_index]
            rc = render_contours[min_index]
            count += 1
            count = count % 5
            e = (vc - rc)[:, np.newaxis]
            cv2.line(contour_image, vc, rc, (128, 128, 128), 1)
            depth = depth_img[rc[1], rc[0]]
            p_camera = np.array([depth * (rc[1] - cx) / fx, depth * (rc[0] - cy) / fy, depth])
            p_model = coord_img[rc[1], rc[0], :]
            inv_z = 1.0 / p_camera[2]
            inv_z2 = inv_z * inv_z
            de_dp_camera = np.array([[-fx * inv_z, 0, fx * p_camera[0] * inv_z2],
                                     [0, -fy * inv_z, fy * p_camera[1] * inv_z2]])  # shape [2, 3]
            de_dtranslation = np.matmul(de_dp_camera, R)  # [2, 3]
            p_model_hat = np.array([[0, -p_model[2], p_model[1]],
                                    [p_model[2], 0, -p_model[0]],
                                    [-p_model[1], p_model[0], 0]])
            de_drotation = -np.matmul(de_dtranslation, p_model_hat)  # [2, 3]
            de_dtheta = np.concatenate([de_drotation, de_dtranslation], axis=1)  # [2, 6]
            # J = de_dtheta, H = J ^ T * J, b = -J ^ T * e
            H += np.matmul(de_dtheta.transpose(), de_dtheta)
            b -= np.matmul(de_dtheta.transpose(), e)

        for vc in visible_contours:
            contour_image[vc[1], vc[0]] = (255, 0, 0)  # blue
        for rc in render_contours:
            contour_image[rc[1], rc[0]] = (0, 0, 255)  # red
        cost = cost / len(visible_contours)
        print(f"iter {it}")
        print(f"cost: {cost}, last cost: {last_cost}")
        if vis_dir == "":
            cv2.namedWindow(f"edge_refine" + str(it) + ".png", cv2.WINDOW_NORMAL)
            cv2.imshow(f"edge_refine" + str(it) + ".png", contour_image)
        else:
            cv2.imwrite(join(vis_dir, "debug" + str(it) + ".png"), contour_image)
        if cost < last_cost:
            last_cost = cost
        else:
            break
        #  Optimize and update pose
        theta = np.linalg.solve(H, b)
        print(f"theta, {theta}")
        dR, _ = cv2.Rodrigues(theta[0:3])
        dt = theta[3:6]
        print(f"dR, {dR}")
        print(f"dt, {dt}")
        t = np.matmul(R, dt) + t
        R = np.matmul(R, dR)

    return R, t, True
