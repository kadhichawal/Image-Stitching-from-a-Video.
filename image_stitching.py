import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn.neighbors import NearestNeighbors
import random
import glob

def extract_video_frames(video_path):
    video = cv2.VideoCapture(video_path) 
    frame_count = 0
    success = True
    
    while success: 
        success, frame = video.read() 
        cv2.imwrite(f"D:\\frames\\frame{frame_count}.jpg", frame) 
        frame_count += 1
        
    return frame_count - 1

def sharpen_image(image):
    sharpen_kernel = np.array([[0, -1, 0], 
                             [-1, 5, -1], 
                             [0, -1, 0]], np.float32)
    
    sharpened = cv2.filter2D(image, -1, kernel=sharpen_kernel)
    return sharpened

def construct_homography_matrix_components(source_pts, target_pts):
    A_matrix = np.array([
        [source_pts[0][0], source_pts[0][1], 1, 0, 0, 0, -source_pts[0][0]*target_pts[0][0], -source_pts[0][1]*target_pts[0][0]],
        [0, 0, 0, source_pts[0][0], source_pts[0][1], 1, -source_pts[0][0]*target_pts[0][1], -source_pts[0][1]*target_pts[0][1]],
        [source_pts[1][0], source_pts[1][1], 1, 0, 0, 0, -source_pts[1][0]*target_pts[1][0], -source_pts[1][1]*target_pts[1][0]],
        [0, 0, 0, source_pts[1][0], source_pts[1][1], 1, -source_pts[1][0]*target_pts[1][1], -source_pts[1][1]*target_pts[1][1]],
        [source_pts[2][0], source_pts[2][1], 1, 0, 0, 0, -source_pts[2][0]*target_pts[2][0], -source_pts[2][1]*target_pts[2][0]],
        [0, 0, 0, source_pts[2][0], source_pts[2][1], 1, -source_pts[2][0]*target_pts[2][1], -source_pts[2][1]*target_pts[2][1]],
        [source_pts[3][0], source_pts[3][1], 1, 0, 0, 0, -source_pts[3][0]*target_pts[3][0], -source_pts[3][1]*target_pts[3][0]],
        [0, 0, 0, source_pts[3][0], source_pts[3][1], 1, -source_pts[3][0]*target_pts[3][1], -source_pts[3][1]*target_pts[3][1]]
    ])
    
    b_vector = np.array([
        [target_pts[0][0]], [target_pts[0][1]],
        [target_pts[1][0]], [target_pts[1][1]],
        [target_pts[2][0]], [target_pts[2][1]],
        [target_pts[3][0]], [target_pts[3][1]]
    ])
    
    return A_matrix, b_vector

def estimate_homography_with_ransac(source_features, target_features, inlier_threshold, max_iterations):
    best_homography = np.zeros((8, 1))
    max_inlier_count = 0
    
    for iteration in range(max_iterations):
        sample_indices = random.sample(range(source_features.shape[0]), 4)
        sampled_source = source_features[sample_indices]
        sampled_target = target_features[sample_indices]
        
        A, b = construct_homography_matrix_components(sampled_source, sampled_target)
        
        try:
            homography = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
            
        homography = np.append(homography, [1]).reshape(3, 3)
        current_inliers = 0
        
        for idx in range(source_features.shape[0]):
            src_point = np.array([source_features[idx][0], source_features[idx][1], 1])
            projected_point = homography @ src_point
            projected_point /= projected_point[2]
            
            error = np.linalg.norm(projected_point[:2] - target_features[idx])
            if error < inlier_threshold:
                current_inliers += 1
                
        if current_inliers > max_inlier_count:
            max_inlier_count = current_inliers
            best_homography = homography
            
    return best_homography

def display_feature_matches(img_base, img_new, base_features, new_features, display_height=500):
    scale_base = display_height / img_base.shape[0]
    scale_new = display_height / img_new.shape[0]
    
    resized_base = cv2.resize(img_base, None, fx=scale_base, fy=scale_base)
    resized_new = cv2.resize(img_new, None, fx=scale_new, fy=scale_new)
    
    adjusted_base = base_features * scale_base
    adjusted_new = new_features * scale_new
    adjusted_new[:, 0] += resized_base.shape[1]
    
    composite_image = np.hstack((resized_base, resized_new))
    
    plt.figure(figsize=(15, 10))
    plt.imshow(composite_image, cmap='gray')
    for pt_base, pt_new in zip(adjusted_base, adjusted_new):
        plt.plot([pt_base[0], pt_new[0]], [pt_base[1], pt_new[1]], 'b-')
        plt.plot(pt_base[0], pt_base[1], 'bo')
        plt.plot(pt_new[0], pt_new[1], 'ro')
    plt.axis('off')
    plt.show()

def detect_and_match_features(img_base, img_new):
    sift = cv2.SIFT_create()
    keypoints_base, descriptors_base = sift.detectAndCompute(img_base, None)
    keypoints_new, descriptors_new = sift.detectAndCompute(img_new, None)
    
    feature_matcher = NearestNeighbors(n_neighbors=2)
    feature_matcher.fit(descriptors_new)
    distances, match_indices = feature_matcher.kneighbors(descriptors_base)
    
    matched_base = []
    matched_new = []
    
    for i, (first_dist, second_dist) in enumerate(distances):
        if first_dist / second_dist < 0.5:
            base_x, base_y = keypoints_base[i].pt
            new_x, new_y = keypoints_new[match_indices[i][0]].pt
            matched_base.append([base_x, base_y])
            matched_new.append([new_x, new_y])
            
    return np.array(matched_base), np.array(matched_new)

def calculate_stitching_canvas_size(base_image, feature_displacements):
    dx = [x1 - x2 for (x1, y1), (x2, y2) in feature_displacements]
    dy = [y1 - y2 for (x1, y1), (x2, y2) in feature_displacements]
    
    canvas = base_image.copy()
    
    # Horizontal expansion
    max_right = math.ceil(max(dx)) if dx else 0
    max_left = abs(math.ceil(min(dx))) if dx else 0
    if max_right > 0:
        canvas = np.hstack((canvas, np.zeros((canvas.shape[0], max_right))))
    else:
        canvas = np.hstack((np.zeros((canvas.shape[0], max_left)), canvas))
    
    # Vertical expansion
    max_down = math.ceil(max(dy)) if dy else 0
    max_up = abs(math.ceil(min(dy))) if dy else 0
    if max_down > 0:
        canvas = np.vstack((canvas, np.zeros((max_down, canvas.shape[1]))))
    else:
        canvas = np.vstack((np.zeros((max_up, canvas.shape[1])), canvas))
    
    return canvas

def stitch_images(base_image, new_image):
    processed_base = sharpen_image(base_image.astype(np.uint8))
    processed_new = sharpen_image(new_image.astype(np.uint8))
    
    base_features, new_features = detect_and_match_features(processed_base, processed_new)
    
    stitching_canvas = calculate_stitching_canvas_size(base_image, zip(base_features, new_features))
    processed_base = sharpen_image(stitching_canvas.astype(np.uint8))
    
    base_features, new_features = detect_and_match_features(processed_base, new_image)
    homography_matrix = estimate_homography_with_ransac(base_features, new_features, 5, 2000)
    
    warped_image = cv2.warpPerspective(new_image, homography_matrix, 
                                      (stitching_canvas.shape[1], stitching_canvas.shape[0]))
    
    mask = (stitching_canvas == 0).astype(np.uint8)
    final_result = stitching_canvas + (warped_image * mask)
    
    return final_result.astype(np.uint8)

if __name__ == '__main__':
    image_paths = glob.glob('./data/*')
    stitched_result = cv2.imread(image_paths[0], cv2.IMREAD_GRAYSCALE)
    
    for idx, img_path in enumerate(image_paths[1:], 1):
        print(f'Processing image {idx}/{len(image_paths)-1}')
        current_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        stitched_result = stitch_images(stitched_result, current_image)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(stitched_result, cmap='gray')
    plt.title('Final Stitched Panorama')
    plt.axis('off')
    plt.show()
