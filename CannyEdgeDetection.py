import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 보조 함수 정의
# 이미지 패딩 함수
def pad_array(array, pad_width, pad_value=0):
    img_arr = np.array(array)
    original_shape = (height, width)

    # original_shape에 pad_width*2를 덧셈한 결과로 튜플 생성
    padded_shape = (original_shape[0] + 2 * pad_width[0], original_shape[1] + 2 * pad_width[1])
    padded_array = np.full(padded_shape, pad_value)

    # 패딩 크기만큼 제외하여 슬라이싱하고, 남은 부분에 배열 복사
    padded_array[pad_width[0]:-pad_width[0], pad_width[1]:-pad_width[1]] = np.mean(img_arr, axis=2)


    # 패딩 처리 된 함수 반환
    return padded_array


# 가우시안 커널 생성
def gaussian_kernel(kernel_size, sigma):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # 모든 요소 합을 계산한 뒤 kernel에서 나눈다
    kernel /= np.sum(kernel)
    return kernel


# 역탄젠트 계산 함수
def arctan(y, x):
    angle = np.zeros_like(y, dtype=np.float64)

    mask1 = x > 0.0
    mask2 = x < 0.0
    mask3 = y < 0.0
    mask4 = ~(mask1 | mask2)  # x == 0인 경우

    angle[mask1] = np.arctan(y[mask1] / x[mask1])
    angle[mask2] = np.arctan(y[mask2] / x[mask2]) + np.pi
    angle[mask3 & mask4] = 3 * np.pi / 2.0
    angle[~(mask3 | mask4)] = np.pi / 2.0

    return angle





# 주요 함수 정의
# 가우시안 필터 처리
def gaussian_filter(image, kernel_size, sigma):
    padded_image = pad_array(image, (kernel_size // 2, kernel_size // 2))
    filtered_image = np.zeros_like(image)

    # 가우시안 커널 생성
    kernel = gaussian_kernel(kernel_size, sigma)

    # 이미지 필터링
    for y in range(height):
        for x in range(width):
            window = padded_image[y : y + kernel_size, x : x + kernel_size]
            filtered_image[y, x] = np.sum(window * kernel)

    return filtered_image


# 소벨 필터 처리
def sobel_filter(img):
    sobel_x = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # 패딩된 이미지를 가져오기 위해 pad_array호출
    padded_image = pad_array(img, (1,1))

    edge_magnitude = np.zeros((height, width), dtype=np.float32)
    edge_direction = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            #현재 픽셀 주변의 3x3을 선택하여 window에 임시 저장
            window = padded_image[y:y+3, x:x+3]

            gradient_x = np.sum(window * sobel_x)
            gradient_y = np.sum(window * sobel_y)

            # 경계선 강도 계산을 위한 각 픽셀 제곱합의 제곱근
            edge_magnitude[y, x] = np.sqrt(gradient_x**2 + gradient_y**2)

            # 경계선 방향 계산
            edge_direction[y, x] = arctan(gradient_y, gradient_x)


    # 찾아낸 에지 반환
    return edge_magnitude, edge_direction


# 비최대 억압 작업
def non_maximum_suppression(gradient_magnitude, gradient_direction):

    # 비최대 억제된 경계선을 저장할 배열 생성
    suppressed_edges = np.zeros((height, width), dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            # 픽셀 방향을 angle에 저장
            angle = gradient_direction[y, x]

            # 경계선의 강도와 주변 픽셀 강도 비교
            if (0 <= angle < 45) or (135 <= angle <= 180):
                q = gradient_magnitude[y, x+1]
                r = gradient_magnitude[y, x-1]
            elif (45 <= angle < 90) or (225 <= angle <= 270):
                q = gradient_magnitude[y-1, x]
                r = gradient_magnitude[y+1, x]
            elif (90 <= angle < 135) or (270 <= angle <= 315):
                q = gradient_magnitude[y-1, x-1]
                r = gradient_magnitude[y+1, x+1]
            else:
                # (135 <= angle < 180) or (315 <= angle <= 360)
                q = gradient_magnitude[y-1, x+1]
                r = gradient_magnitude[y+1, x-1]

            # 현재 픽셀의 강도와 주변 픽셀 강도 비교하여 비최대 억제
            if (gradient_magnitude[y, x] >= q) and (gradient_magnitude[y, x] >= r):
                suppressed_edges[y, x] = gradient_magnitude[y, x]

    return suppressed_edges


# 이중 임계값 처리
def double_threshold(image, minVal, maxVal):

    # 결과를 저장할 배열 생성
    result = np.zeros_like(image)

    # 강한 엣지 설정
    strong_edge_value = 255

    # 약한 엣지 설정
    weak_edge_value = 50

    # 이미지를 반복하여 픽셀을 검사하고 엣지를 결정
    for y in range(height):
        for x in range(width):
            pixel = image[y, x]

            if pixel >= minVal:
                # 강한 엣지로 설정
                result[y, x] = strong_edge_value
            elif pixel >= maxVal:
                # 약한 엣지로 설정
                result[y, x] = weak_edge_value

    return result


# hysteresis 추적
def edge_tracking_by_hysteresis(edge_map, minVal, maxVal):

    # dfs를 이용해 각 배열을 순회
    # 방문을 나타내기 위한 boolean타입의 visited배열
    visited = np.zeros((height, width), dtype=bool)


    # Depth First Search
    def dfs(y, x):

        # 배열 크기를 벗어난 경우 종료
        if y < 0 or y >= height or x < 0 or x >= width:
            return

        # 이미 방문을 했거나 min보다 작으면 종료
        if visited[y, x] or edge_map[y, x] < minVal:
            return

        visited[y, x] = True
        edge_map[y, x] = 255

        # 이웃에 있는 픽셀 탐색하여 강한 엣지와 연결된 경계를 찾는 반복문+재귀문
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                dfs(ny, nx)


    # 방문하지 않은 것 중, 강한 에지와 연결된 경계를 찾는 반복문
    for y in range(height):
        for x in range(width):
            if not visited[y, x] and edge_map[y, x] >= maxVal:
                dfs(y, x)


    edge_map[edge_map < 255] = 0

    return edge_map





# 이미지 불러오기
img = Image.open('test_image.jpg').convert('RGB')
width, height = img.size


kernel_size = 5
sigma = 1.0




# 1. Noise Reduction 노이즈 제거
gaussian_array = gaussian_filter(img, kernel_size, sigma)
gaussian_img = Image.fromarray(gaussian_array)


# 2. Gradient Calculation 기울기 계산
gradient_magnitude, gradient_direction = sobel_filter(gaussian_img)
gradient_img = Image.fromarray(gradient_magnitude)


# 3. Non-maximum Suppression 비최대 억압
line_array = non_maximum_suppression(gradient_magnitude, gradient_direction)
line_img = Image.fromarray(line_array)


# 4. Double Threshold
thresholding = double_threshold(line_array, 50, 150)
threshold_img = Image.fromarray(thresholding)


# 5. Edge Tracking by Hysteresis
hysteresis = edge_tracking_by_hysteresis(thresholding, 50, 150)
hysteresis_img = Image.fromarray(hysteresis)





# 이미지 출력
plt.figure(figsize=(14, 6))
plt.suptitle('Canny Edge Detection', fontsize=20)

plt.subplot(2,3,1)
plt.imshow(img)
plt.title("Original Image")

plt.subplot(2,3,2)
plt.imshow(gaussian_img)
plt.title("Gaussian")

plt.subplot(2,3,3)
plt.imshow(gradient_img)
plt.title("Sobel")

plt.subplot(2,3,4)
plt.imshow(line_img)
plt.title("Non-maximum")

plt.subplot(2,3,5)
plt.imshow(threshold_img)
plt.title("Double Threshold")

plt.subplot(2,3,6)
plt.imshow(hysteresis_img)
plt.title("Edge Tracking by Hysteresis")

plt.tight_layout()
plt.show()
