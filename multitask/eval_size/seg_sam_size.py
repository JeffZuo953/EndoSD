import numpy as np
import cv2
import os
import time


def backproject_with_color(depth: np.ndarray, image: np.ndarray, mask: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    将深度图和图像 backproject 成 3D 点和颜色
    返回：
        - pts: N×3 的 XYZ 点
        - colors: N×3 的 RGB 值，值域 [0,1]
    """
    h, w = depth.shape
    v, u = np.where(mask > 0)
    # z = depth[v, u] / 1000.0
    z = depth[v, u]  # meters

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    colors = image[v, u].astype(np.float32) / 255.0  # 转为 float RGB
    return pts, colors


def pca_lengths(points: np.ndarray) -> tuple[float, float, float]:
    """
    Principal‑component lengths (L1≥L2≥L3) for Nx3 point cloud.
    Returns lengths in the same units as input (millimetres).
    """
    pts = points - points.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    order = eigval.argsort()[::-1]
    eigvec = eigvec[:, order]  # columns are e1,e2,e3
    proj = pts @ eigvec
    lengths = proj.max(axis=0) - proj.min(axis=0)
    return tuple(lengths)


def mask2size2(img_ori, semantic_mask, depth_map, sorted_stone_mask, save_gray_img, save_add_img):
    v_0 = 186.1156
    f_v = 252.5626  # todo change camera intrinsic
    fx, fy, cx, cy = f_v, f_v, v_0, v_0

    depth_map = depth_map.astype(np.float32) / 255.0 * 0.025  # todo metric depth, change max-depth

    if len(sorted_stone_mask) == 0:
        print('no stone_mask')
        save_img = cv2.cvtColor(semantic_mask, cv2.COLOR_GRAY2BGR)
        return save_img, img_ori
    else:
        save_image = cv2.cvtColor(np.zeros_like(sorted_stone_mask, dtype=np.uint8), cv2.COLOR_GRAY2BGR) if save_gray_img else None
        stone_size_max = 0
        for [stone_area, s_msk_img] in sorted_stone_mask:
            # t_7 = time.time()
            contour_stone, _ = cv2.findContours(s_msk_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_stone = max(contour_stone, key=cv2.contourArea)
            rect_stone = cv2.minAreaRect(contour_stone)

            t1 = time.time()
            pts, colors = backproject_with_color(depth_map, img_ori, s_msk_img, fx, fy, cx, cy)
            print(f"[PointCloud] {pts.shape[0]} points with color", time.time() - t1)

            t2 = time.time()
            L1, L2, L3 = pca_lengths(pts)
            Lmax = max(L1, L2, L3) * 1000  # mm
            print(f"[Dimensions] Lmax={Lmax:.2f} mm, L1={L1 * 1000:.2f} mm, "
                  f"L2={L2 * 1000:.2f} mm, L3={L3 * 1000:.2f} mm",
                  time.time() - t2)

            if stone_size_max < Lmax:
                stone_size_max = Lmax

            if save_gray_img:
                save_image[s_msk_img > 0] = [255, 255, 255]
                cv2.drawContours(save_image, [contour_stone], -1, (0, 0, 255), 2)
                cv2.putText(save_image, '%.2f' % Lmax, (int(rect_stone[0][0]) - 10, int(rect_stone[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 200), 1)

            if save_add_img:
                cv2.drawContours(img_ori, [contour_stone], -1, (0, 0, 255), 2)
                cv2.putText(img_ori, '%.2f' % Lmax, (int(rect_stone[0][0]) - 10, int(rect_stone[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            # print("t8", time.time() - t_8)

        t_9 = time.time()

        if save_gray_img:
            cv2.putText(save_image, 'stone_max: ' + '%.2f' % stone_size_max, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 255), 1)

        if save_add_img:
            cv2.putText(img_ori, 'stone_max (mm): ' + '%.2f' % stone_size_max, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 255), 1)

        # print("t9", time.time() - t_9)
        print('stone_size_max', stone_size_max)

        return save_image, img_ori


if __name__ == '__main__':

    save_gray_img, save_add_img = False, True  # save size result: grayscale/add img_ori
    size_gray2, size_img2 = mask2size2(img_img, semantic_mask, depth_map, stone_mask, save_gray_img, save_add_img)
