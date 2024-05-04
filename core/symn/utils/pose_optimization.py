from os.path import join
import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2
try:
    import open3d as o3d
except:
    pass
from scipy.optimize import minimize
from .class_id_encoder_decoder import RGB_to_class_id, class_id_to_class_code_images


class Rodrigues(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rvec):
        R, jac = cv2.Rodrigues(rvec.detach().cpu().numpy())
        jac = torch.from_numpy(jac).to(rvec.device)
        ctx.save_for_backward(jac)
        return torch.from_numpy(R).to(rvec.device)

    @staticmethod
    def backward(ctx, grad_output):
        (jac,) = ctx.saved_tensors
        return jac @ grad_output.to(jac.device).reshape(-1)


def test_Rodrigues():
    # run this function to check user's function "Rodrigues" is correct
    test_input = torch.randn((3,), requires_grad=True, dtype=torch.float64)
    torch.autograd.gradcheck(Rodrigues.apply, (test_input,), eps=1e-6)


def code_renderer(meta_info, obj_id):
    model_GT_path = join(
        meta_info.models_GT_color_folder, "obj_{:06d}.ply".format(obj_id)
    )
    pcd = o3d.io.read_point_cloud(str(model_GT_path))
    color = np.array(pcd.colors)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    return pcd_tree, color


def pose_optimization(
    R: np.ndarray,
    t: np.ndarray,
    K,
    obj_id,
    amodal_mask_prob,
    visib_mask_prob,
    binary_code_prob,
    renderer,
    pcd_tree,
    color,
    method="BFGS",
):
    """
    Optimize estimated pose (R, t) by local maximization of the code align
    :param R: np.ndarray (3, 3)
    :param t: np.ndarray (3, )
    :param K: np.ndarray (3, 3)
    :param amodal_mask_prob: torch.Tensor (128, 128)
    :param visib_mask_prob: torch.Tensor (128, 128)
    :param binary_code_prob: torch.Tensor (16, 128, 128)
    :param renderer:
    :param pcd_tree:
    :param color:
    :param method:
    :return: opti_R, opti_t
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if len(t.shape) == 1:  # convert [3,] to  [3, 1]
        t = t[:, np.newaxis]
    device = binary_code_prob.device
    coord_img = renderer.render(obj_id, K, R, t)
    UV = np.nonzero(coord_img[..., 3])
    coord_img = coord_img[..., :3]
    coord_img = renderer.denormalize(coord_img, obj_id)
    coordinates = coord_img[UV]  # [N, 3]
    labels = np.zeros_like(coord_img)
    for coordi, u, v in zip(coordinates, UV[0], UV[1]):
        k, idx, _ = pcd_tree.search_knn_vector_3d(coordi, 1)
        labels[u, v] = color[idx] * 255.0
    class_id_image = RGB_to_class_id(labels)
    code_images = class_id_to_class_code_images(class_id_image)  # [H , W, 16]
    code_masked = torch.from_numpy(code_images[UV]).to(device).float()  # [N, 16]
    coordinates = torch.cat(
        (
            torch.from_numpy(coordinates).to(device),
            torch.ones(len(coordinates), 1, device=device),
        ),
        dim=1,
    ).float()  # [N, 4]
    K = torch.from_numpy(K).to(device)
    # show_code("render code", preprogress_code(code_images))
    # show_code("predict code", preprogress_code(binary_code_prob))
    # show_mask("visible mask", preprogress_mask(visib_mask_prob))

    def sample(img, p_img_norm):
        samples = F.grid_sample(
            img.permute(2, 0, 1)[None],  # (1, d, H, W)
            p_img_norm[None, None],  # (1, 1, N, 2)
            align_corners=False,
            padding_mode="border",
            mode="bilinear",
        )  # (1, d, 1, N)
        return samples[0, :, 0].T  # (N, d)

    def objective(pose: np.ndarray, return_grad=False):
        pose = torch.from_numpy(pose).float()
        pose.requires_grad = return_grad
        Rt = torch.cat(
            (
                Rodrigues.apply(pose[:3]),
                pose[3:, None],
            ),
            dim=1,
        ).to(
            device
        )  # (3, 4)
        P = K @ Rt
        p_img = coordinates @ P.T
        p_img = p_img[..., :2] / p_img[..., 2:]  # (N, 2)
        # pytorch grid_sample coordinates
        p_img_norm = (p_img + 0.5) * (2 / 128) - 1
        code_prob_sampled = sample(
            binary_code_prob.permute(1, 2, 0), p_img_norm
        )  # (N, 16)
        score = (code_prob_sampled - 2 * code_prob_sampled * code_masked).mean()

        if return_grad:
            score.backward()
            return pose.grad.detach().cpu().numpy()
        else:
            return score.item()

    rvec = cv2.Rodrigues(R)[0]
    pose = np.array((*rvec[:, 0], *t[:, 0]))
    result = minimize(
        fun=objective,
        x0=pose,
        jac=lambda pose: objective(pose, return_grad=True),
        method=method,
    )
    pose = result.x
    R = cv2.Rodrigues(pose[:3])[0]
    t = pose[3:]
    return R, t, result.fun


def pose_pnp(K, visib_mask_prob, binary_code_prob, decoder):
    """
    Optimize estimated pose (R, t) by PnP method from openCV
    :param K: np.ndarray (3, 3)
    :param visib_mask_prob: torch.Tensor (128, 128)
    :param binary_code_prob: torch.Tensor (16, 128, 128)

    :return: est_R, est_t, success
    """
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    # process visib_mask
    visib_mask = (visib_mask_prob > 0.5).int().detach().cpu().numpy()
    # process binary_code
    binary_code = (binary_code_prob > 0.5).int().detach().cpu().numpy()
    while len(binary_code.shape) > 3:  # convert [1, 16, h, w] to [16, h, w]
        binary_code = binary_code[0]
    class_id_image = np.zeros((binary_code.shape[1], binary_code.shape[2]), dtype=int)
    codes_length = binary_code.shape[0]
    for i in range(codes_length):
        class_id_image += binary_code[i, ...] * (2 ** (16 - 1 - i))

    visib_points = np.transpose(visib_mask.nonzero())  # (npoint, 2)
    point_2D = np.zeros((visib_points.shape[0], 2))
    point_3D = np.zeros((visib_points.shape[0], 3))
    valid_point = 0
    for point in visib_points:
        id_for_searching = class_id_image[point[1], point[0]]
        if len(decoder[id_for_searching]):
            point_2D[valid_point] = point
            point_3D[valid_point] = random.choice(decoder[id_for_searching])
            valid_point += 1
    point_2D = point_2D[:valid_point]
    point_3D = point_3D[:valid_point]
    if len(point_3D) < 6:
        return None, None, False
    _, rvecs, tvecs, inliers = cv2.solvePnPRansac(
        point_3D.astype(np.float32),
        point_2D.astype(np.float32),
        K,
        distCoeffs=None,
        reprojectionError=2,
        iterationsCount=150,
        flags=cv2.SOLVEPNP_EPNP,
    )
    rot = cv2.Rodrigues(rvecs, jacobian=None)[0]
    if np.any(np.isnan(tvecs)):
        return None, None, False
    return rot, tvecs.squeeze(), True


if __name__ == "__main__":
    test_Rodrigues()