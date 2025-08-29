import cv2
import os
import pandas as pd
import numpy as np
import cupy as cp


def load_and_flip_odds(folder_path):
    img_dict = {}

    for filename in os.listdir(folder_path):
        # Read each pic
        id = int(filename[3:6])
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, 0) # greyscale, shape (height, width)

        # Flip odd pic
        if id%2 == 1:
            img = cv2.flip(img, 1)

        img_dict[filename[:6]] = img

    return img_dict


def crop_black_edge(img_dict):
    crop_img_dict = {}

    for name, img in img_dict.items():
        # Find the left boundary directly on the original image
        vertical_sum = np.sum(img>0, axis=0)  # sums each column, non-black pixels to 1
        left_bound = np.argmax(vertical_sum>1) # when the first col has more than 1 non-black pixel, get the left boundary

        # Crop the image
        img_crop = img[:, left_bound:-left_bound]

        crop_img_dict[name] = img_crop

    return crop_img_dict


def adaptive_median_filter_numpy(img_dict): # numpy implemention is slow, use adaptive_median_filter function (cupy version) in training
    filter_img_dict = {}

    for name, img in img_dict.items():
        rows, cols = img.shape

        # Calculate noise density E: the fraction of pixels that are noisy (0 and 255)
        noisy_pixels = np.logical_or(img == 0, img == 255)
        total_pixels = rows*cols
        E = np.sum(noisy_pixels)/total_pixels
        
        output = img.copy()
        
        for i1 in range(rows):
            for i2 in range(cols):
                pixel = output[i1, i2]

                # None-noisy pixel: remained the same
                if pixel != 0 and pixel != 255:
                    continue
                
                # Noisy pixel: start adaptive filtering
                W = 3 # start with 3*3 window
                while True:
                    r = (W - 1)//2 # radius: distance from center pixel to edge
                    S = [] # list of candidate pixels for calculating to replace the current pixel 

                    if E < 0.5: # for low noise, use only 4 endpoints, this makes calculation faster
                        positions = [
                            (i1, i2 - r), # left
                            (i1, i2 + r), # right
                            (i1 - r, i2), # up
                            (i1 + r, i2) # down
                        ]
                        for p1, p2 in positions:
                            if 0 <= p1 < rows and 0 <= p2 < cols: # make sure endpoints are inside the img
                                S.append(img[p1, p2])
                    
                    else: # for high noise, use all-around endpoints
                        for j1 in range(i1 - r, i1 + r + 1): # left column
                            p1, p2 = j1, i2 - r
                            if 0 <= p1 < rows and 0 <= p2 < cols:
                                S.append(img[p1, p2])
                        for j1 in range(i1 - r, i1 + r + 1): # right column
                            p1, p2 = j1, i2 + r
                            if 0 <= p1 < rows and 0 <= p2 < cols:
                                S.append(img[p1, p2])
                        for j2 in range(i2 - r, i2 + r + 1): # top row
                            p1, p2 = i1 - r, j2
                            if 0 <= p1 < rows and 0 <= p2 < cols:
                                S.append(img[p1, p2])
                        for j2 in range(i2 - r, i2 + r + 1): # bottom row
                            p1, p2 = i1 + r, j2
                            if 0 <= p1 < rows and 0 <= p2 < cols:
                                S.append(img[p1, p2])
                    
                    # In the candidate pixels, use only non-noisy ones
                    H = [p for p in S if p != 0 and p != 255]
                    if len(H) > 0:
                        output[i1, i2] = np.median(H)
                        break
                    else: # expand window by 1 pixel in all directions when no non-noisy candidates
                        W += 2
        
        filter_img_dict[name] = output

    return filter_img_dict


def adaptive_median_filter(img_dict): # cupy implemention with parallelism in alogrithms
    adaptive_median_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void adaptive_median(const unsigned char* img, unsigned char* out, int rows, int cols, float E) {
            int i1 = blockDim.y * blockIdx.y + threadIdx.y;
            int i2 = blockDim.x * blockIdx.x + threadIdx.x;

            if (i1 >= rows || i2 >= cols) return;

            unsigned char pixel = img[i1 * cols + i2];

            if (pixel != 0 && pixel != 255) {
                out[i1 * cols + i2] = pixel;
                return;
            }

            int W = 3;
            while (1) {
                int r = (W - 1) / 2;
                unsigned char S[64]; // max candidate pixels, enough for 7x7
                int count = 0;

                if (E < 0.5f) {
                    // 4 endpoints
                    int positions[4][2] = {{i1, i2 - r}, {i1, i2 + r}, {i1 - r, i2}, {i1 + r, i2}};
                    for (int k = 0; k < 4; ++k) {
                        int x = positions[k][0];
                        int y = positions[k][1];
                        if (x >= 0 && x < rows && y >= 0 && y < cols) {
                            unsigned char val = img[x * cols + y];
                            S[count++] = val;
                        }
                    }
                } else {
                    // full edges
                    for (int j1 = i1 - r; j1 <= i1 + r; ++j1) {
                        if (j1 >= 0 && j1 < rows) {
                            if (i2 - r >= 0) S[count++] = img[j1 * cols + (i2 - r)];
                            if (i2 + r < cols) S[count++] = img[j1 * cols + (i2 + r)];
                        }
                    }
                    for (int j2 = i2 - r; j2 <= i2 + r; ++j2) {
                        if (j2 >= 0 && j2 < cols) {
                            if (i1 - r >= 0) S[count++] = img[(i1 - r) * cols + j2];
                            if (i1 + r < rows) S[count++] = img[(i1 + r) * cols + j2];
                        }
                    }
                }

                // median of non-noisy
                unsigned char H[64];
                int hcount = 0;
                for (int k = 0; k < count; ++k) {
                    if (S[k] != 0 && S[k] != 255) H[hcount++] = S[k];
                }

                if (hcount > 0) {
                    // simple insertion sort to get median
                    for (int m = 1; m < hcount; ++m) {
                        unsigned char key = H[m];
                        int n = m - 1;
                        while (n >= 0 && H[n] > key) {
                            H[n + 1] = H[n];
                            n--;
                        }
                        H[n + 1] = key;
                    }
                    out[i1 * cols + i2] = H[hcount / 2];
                    break;
                } else {
                    W += 2; // expand
                    if (W > 15) { // safety limit
                        out[i1 * cols + i2] = pixel;
                        break;
                    }
                }
            }
        }
        ''', 'adaptive_median'
        ) # define this kernel for later parallel processing of all pixels
    filter_img_dict = {}

    for name, img in img_dict.items():
        img_gpu = cp.asarray(img, dtype=cp.uint8)
        out_gpu = cp.empty_like(img_gpu)

        rows, cols = img_gpu.shape
        E = float(cp.sum((img_gpu == 0) | (img_gpu == 255))/(rows * cols))

        threads_per_block = (16, 16)
        blocks_x = (cols + threads_per_block[0] - 1)//threads_per_block[0]
        blocks_y = (rows + threads_per_block[1] - 1)//threads_per_block[1]

        adaptive_median_kernel((blocks_x, blocks_y), threads_per_block,
                               (img_gpu, out_gpu, rows, cols, E))

        filter_img_dict[name] = cp.asnumpy(out_gpu)

    return filter_img_dict


def CLAHE(img_dict): # Contrast Limited Adaptive Histogram Equalization
    clahe_img_dict = {}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    for name, img in img_dict.items():
        clahe_img = clahe.apply(img)
        clahe_img_dict[name] = clahe_img

    return clahe_img_dict


def edge_detection(img_dict): # canny edge detection with sober filter
    edge_dict = {}

    for name, img in img_dict.items():
        # Blur a little so tiny noise doesn't trick into regarded as an edge. This is a common step before edge detection. 
        blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=1.8) # for ksize=(0, 0), cv2 will set the kernel size accroding to sigma

        # Canny edge detection
        edge = cv2.Canny(blurred,
                          threshold1=30, threshold2=50, # double Thresholding for strong-weak gradient, middle gradient is applied edge tracking by hysteresis
                          L2gradient=True
                          ) # Note that cv2 automately thins the edges with non-maximum suppression to keep the sharpest middle part of each edge (1-pixel edge).
        
        edge_dict[name] = edge
    
    return edge_dict


def line_detection(edge_dict, img_dict): # detect the seperate line of PM (pectoral muscle) and breast tissue part
    line_dict = {}
    
    for name, edge in edge_dict.items():
        # Hough Transform to distract all lines
        all_lines = cv2.HoughLines(image=edge,
                               rho=5, # tolerant how n pixel far off the line
                               theta=np.pi/180, # each degree is checked
                               threshold=50 # at least n pixles to be a line
                               )
        
        # Filter lines based on angle and distance
        candidate_lines = []
        for line in all_lines:
            rho, theta = line[0]
            angle_deg = np.degrees(theta)
            if 30 <= angle_deg <= 60 and 50 <= rho <= 300:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*a)
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*a)
                candidate_lines.append((x1, y1, x2, y2))

        # Select line with minimal information loss
        best_line = None
        max_intensity_sum = 0
        for x1, y1, x2, y2 in candidate_lines:
            mask = np.zeros_like(img_dict[name])
            cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=3) # set line of n thickness as white 255, leave non-line pixels black 0
            temp_img = img_dict[name].copy()
            temp_img[mask==255] = 0 # set line in the original img black 0
            intensity_sum = temp_img.sum()
            if intensity_sum > max_intensity_sum: # the remaining pixels' value sum highest means minimal info loss
                max_intensity_sum = intensity_sum
                best_line = (x1, y1, x2, y2)
        line_dict[name] = best_line

    return line_dict


def remove_pm(line_dict, img_dict):
    nopm_img_dict = {}

    for name, img in img_dict.items():
        if line_dict[name] == None: # !!! Note: this is to test whole workflow going. Delete this part when training to make sure Line_dict is complete (i.e., make sure all imgs got lines). 
            continue
        
        nopm_img = img.copy()
        x1, y1, x2, y2 = line_dict[name]

        # Define polygon points: top-left corner + the line endpoints
        polygon = np.array([[(0, 0), (x1, y1), (x2, y2)]], dtype=np.int32)

        # Create mask
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.fillPoly(mask, polygon, 255)

        # Zero out pixels inside the polygon
        nopm_img[mask==255] = 0

        nopm_img_dict[name]=nopm_img
        
    return nopm_img_dict
    

def aug(img_dict):
    aug_img_dict = {}
    
    for name, img in img_dict.items():
        aug_img_dict[name] = img

        # Rotations (45, 90, 135, 180, 235, 270)
        for angle in [45, 90, 135, 180, 235, 270]:
            h, w = img.shape
            center = (w//2, h//2)
            matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
            rotated = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR) # apply rotation to img with same output size and interpolation method for filling
            aug_img_dict[f'{name}_{angle}'] = rotated

        # Horizontal and vertical flip
        aug_img_dict[f'{name}_hori'] = cv2.flip(img, 1)
        aug_img_dict[f'{name}_verti'] = cv2.flip(img, 0)

    return aug_img_dict


def get_label_dict(csv_path, aug_img_dict):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Apply severity priority and keep only one row per img
    label_map = {"Malignant": 0, "Benign": 1, "Normal": 2}
    df["label"] = df["SEVERITY"].map(label_map)
    df = df.sort_values("label")
    df = df.drop_duplicates("REFNUM", keep="first") # keep only one row in order: Malignant > Benign > Normal

    # Build labels_dict from keys in augmented_img_dict
    label_dict = {}
    for name in aug_img_dict.keys():
        basic_name = name.split("_")[0]   # strip augmentation suffix and keep only the first like mdb001
        label_dict[name] = int(df.loc[df['REFNUM'] == basic_name, 'label'].iloc[0])

    return label_dict