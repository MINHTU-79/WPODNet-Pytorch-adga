# Nội dung chính của trang web GitHub.dev
# Trang web này hiển thị một dự án phát triển phần mềm có tên là "WPODNet-Pytorch-adga" đang được mở trong môi trường editor GitHub.dev. Khu vực nội dung chính đang hiển thị mã nguồn của file backend.py trong dự án này.

# Chi tiết cụ thể như sau:

# 	1. Tập trung vào file backend.py: Đây là file Python đang được chỉnh sửa, có chứa mã nguồn liên quan đến xử lý dự đoán (prediction) cho mô hình WPODNet.

# 	2. Nội dung mã nguồn file này bao gồm:


# 		* Các import cần thiết từ các thư viện Python như typing, numpy, torch, PIL, và torchvision
# 		* Import lớp WPODNet từ module .model trong cùng dự án
# 		* Định nghĩa một lớp Prediction có các phương thức:

# 			* __init__: Khởi tạo đối tượng với hình ảnh, ranh giới và độ tin cậy
# 			* _get_perspective_coeffs: Tính toán hệ số phối cảnh dựa trên ranh giới
# 			* annotate: Vẽ đường viền quanh vùng được nhận diện trên hình ảnh
# 			* warp: Dường như là phương thức chuẩn bị được triển khai (chưa hoàn thiện)
# 	3. Chi tiết kỹ thuật:


# 		* Lớp Prediction xử lý kết quả từ một mô hình nhận diện biển số xe (dựa trên tên dự án "WPODNet")
# 		* Có khả năng chuyển đổi tọa độ từ ranh giới đã phát hiện thành hệ số phối cảnh
# 		* Cung cấp công cụ để vẽ chú thích trên hình ảnh gốc, đánh dấu vùng biển số được phát hiện
# 	4. Bối cảnh dự án rộng hơn:


# 		* WPODNet là một mạng neural được sử dụng để phát hiện biển số xe trong hình ảnh
# 		* Đây có thể là một phiên bản PyTorch được điều chỉnh (adaptation) của mô hình WPODNet, như được gợi ý bởi tên dự án "WPODNet-Pytorch-adga"

# Mã nguồn này là một phần của một hệ thống lớn hơn để xử lý và nhận dạng biển số xe từ hình ảnh, sử dụng các kỹ thuật thị giác máy tính và học sâu.

from typing import List, Tuple

	# * typing là một module chuẩn của Python, cung cấp các công cụ để khai báo kiểu dữ liệu trong mã Python.
	# * List và Tuple là các kiểu dữ liệu tập hợp (collection types) được sử dụng trong type hints.
	# * Dòng này import hai type hints để sử dụng cho việc khai báo kiểu dữ liệu.

import numpy as np
import torch
from PIL import Image, ImageDraw

# Giải thích:
# 	* PIL (Python Imaging Library) còn được gọi là Pillow, là thư viện xử lý ảnh cơ bản trong Python.
# 	* Image là lớp chính để đọc, xử lý và lưu hình ảnh.
# 	* ImageDraw cung cấp các công cụ để vẽ hình dạng (đường, hình chữ nhật, v.v.) lên hình ảnh.

# Chi tiết kỹ thuật:
# 	* Image cung cấp các phương thức để đọc, ghi và xử lý ảnh với nhiều định dạng khác nhau (JPG, PNG, v.v.)
# 	* ImageDraw cho phép thêm đồ họa, văn bản, và các phần tử khác vào ảnh
# 	* PIL làm việc tốt với numpy và có thể chuyển đổi qua lại giữa đối tượng Image và mảng numpy
from torchvision.transforms.functional import (_get_perspective_coeffs,
                                               to_tensor)


# Giải thích:
# torchvision là một gói thư viện bổ sung cho PyTorch, tập trung vào xử lý hình ảnh và thị giác máy tính.
# transforms.functional chứa các hàm xử lý và biến đổi ảnh dưới dạng hàm thuần túy (không phải đối tượng).
# _get_perspective_coeffs là một hàm nội bộ tính toán ma trận biến đổi phối cảnh (perspective transformation).
# to_tensor là hàm chuyển đổi hình ảnh PIL thành tensor PyTorch.
# Chi tiết kỹ thuật:
# _get_perspective_coeffs tính toán ma trận biến đổi từ 4 điểm nguồn sang 4 điểm đích
# to_tensor thực hiện:
# Chuyển đổi PIL Image hoặc numpy.ndarray sang tensor PyTorch
# Điều chỉnh kênh màu từ (H x W x C) sang (C x H x W)
# Chuẩn hóa giá trị pixel từ [0, 255] sang [0.0, 1.0]


# 2. Ma trận biến đổi phối cảnh là gì?
# Biến đổi phối cảnh (perspective transformation) là một kỹ thuật xử lý ảnh cho phép:

# 	* Thay đổi góc nhìn của một đối tượng trong ảnh
# 	* "Làm thẳng" các đối tượng bị chụp từ góc nghiêng
# 	* Chỉnh sửa hình dạng của một vùng ảnh từ tứ giác bất kỳ thành hình chữ nhật tiêu chuẩn

# Ma trận biến đổi phối cảnh là một ma trận 3x3 chứa các hệ số ánh xạ điểm từ ảnh nguồn sang ảnh đích.

from .model import WPODNet

# một lớp được thiết kế để xử lý và lưu trữ kết quả nhận dạng biển số xe. 
# Lớp này đóng gói dữ liệu về hình ảnh biển số, vị trí của nó trong ảnh gốc, và độ tin cậy của kết quả dự đoán.

class Prediction:
    def __init__(self, image: Image.Image, bounds: np.ndarray, confidence: float):
        self.image = image
        self.bounds = bounds
        self.confidence = confidence

    # * image: Image.Image: Một đối tượng hình ảnh từ thư viện PIL (Python Imaging Library). Đây là hình ảnh gốc nơi biển số được phát hiện.

	# * bounds: np.ndarray: Một mảng numpy chứa tọa độ các góc của biển số xe. Thông thường có dạng mảng 2D với kích thước (4,2), mỗi hàng chứa tọa độ (x,y) của một góc biển số.

	# * confidence: float: Một số thực từ 0 đến 1 thể hiện độ tin cậy của dự đoán, với 1 là hoàn toàn chắc chắn.


    # a) Các tham số đầu vào:
    # self: Đối tượng hiện tại (instance của lớp Prediction)
    # width: int: Chiều rộng mong muốn của hình ảnh biển số sau khi biến đổi (thường là 208px)
    # height: int: Chiều cao mong muốn của hình ảnh biển số sau khi biến đổi (thường là 60px)
    # b) Các biến trong phương thức:
    # src_points: Tọa độ của 4 điểm góc của biển số trong hình ảnh gốc (chuyển từ numpy array sang list)
    # dst_points: Tọa độ mới của 4 điểm góc trong hình ảnh đích sau khi biến đổi
    # c) Kết quả trả về:
    # Một danh sách các hệ số phối cảnh dùng cho phép biến đổi hình ảnh


    def _get_perspective_coeffs(self, width: int, height: int) -> List[float]:
        # Get the perspective matrix
        src_points = self.bounds.tolist()
        dst_points = [[0, 0], [width, 0], [width, height], [0, height]]
        return _get_perspective_coeffs(src_points, dst_points)


	# * Mục đích: Phương thức này dùng để vẽ đường viền quanh biển số xe đã phát hiện được, giúp đánh dấu vị trí biển số trong ảnh gốc.
	# * Tham số:
 
	# 	* self: Tham chiếu đến đối tượng hiện tại (instance của lớp Prediction), chứa thông tin về biển số đã phát hiện.
	# 	* outline: Tham số kiểu chuỗi (str), mặc định là 'red', xác định màu của đường viền quanh biển số.
	# 	* width: Tham số kiểu số nguyên (int), mặc định là 3, xác định độ dày của đường viền.
	# * Kiểu trả về: Image.Image - một đối tượng hình ảnh từ thư viện Pillow (PIL).

    def annotate(self, outline: str = 'red', width: int = 3) -> Image.Image:
        canvas = self.image.copy()
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(
            [(x, y) for x, y in self.bounds],
            outline=outline,
            width=width
        )
        return canvas

    # * self.image: Hình ảnh gốc chứa biển số xe
	# * .transform(): Phương thức của đối tượng Image trong thư viện PIL để thực hiện biến đổi hình ảnh
	# * (width, height): Tuple chỉ định kích thước đầu ra của hình ảnh sau khi biến đổi
	# * Image.PERSPECTIVE: Hằng số chỉ định loại biến đổi là biến đổi phối cảnh
	# * coeffs: Các hệ số biến đổi đã tính toán ở bước trước
	# * Kết quả được lưu trong biến warped - hình ảnh sau khi đã được "làm phẳng"

    def warp(self, width: int = 208, height: int = 60) -> Image.Image:
        # Get the perspective matrix
        coeffs = self._get_perspective_coeffs(width, height)
        warped = self.image.transform((width, height), Image.PERSPECTIVE, coeffs)
        return warped

    # Lớp Predictor có chức năng chính là phát hiện và định vị biển số xe trong hình ảnh. Cụ thể:

    #     1. Mục đích chính: Nhận diện vị trí biển số xe từ ảnh đầu vào và xác định chính xác tọa độ 4 góc của biển số.

    #     2. Đầu vào: Hình ảnh chụp xe (có thể là từ camera giám sát, camera bãi đỗ xe, v.v.)

    #     3. Đầu ra: Đối tượng Prediction chứa:


    #         * Ảnh gốc
    #         * Tọa độ 4 góc của biển số xe
    #         * Độ tin cậy của phát hiện
    #     4. Điểm mạnh: Không chỉ xác định vị trí biển số mà còn tính toán ma trận biến đổi hình học để xử lý biển số bị nghiêng, bị biến dạng do góc chụp.


    # Ví dụ đơn giản: Khi bạn có một ảnh chụp ô tô, lớp Predictor sẽ tìm và đánh dấu chính xác vị trí biển số xe trong ảnh đó, kể cả khi biển số bị nghiêng hoặc chụp từ góc không thẳng.

    # Đây là bước đầu tiên trong quy trình nhận dạng biển số xe tự động, trước khi thực hiện các bước tiếp theo như cắt, làm phẳng và nhận dạng ký tự trên biển số.

# ================


    # Ứng dụng cụ thể trong quy trình xử lý:
    # 	1. Chụp ảnh ô tô: Bạn có một ảnh chụp ô tô từ camera an ninh với góc nghiêng.

    # 	2. Phát hiện biển số: Mô hình WPODNet quét ảnh và tìm ra khu vực có khả năng chứa biển số xe.

    # 	3. Dự đoán ma trận biến đổi: WPODNet dự đoán ma trận biến đổi từ biển số chuẩn (đại diện bởi _q) đến biển số trong ảnh thực tế.

    # 	4. Áp dụng ma trận biến đổi: Nhân ma trận biến đổi với _q để xác định chính xác vị trí 4 góc của biển số trong ảnh.

    # 	5. Biến đổi ngược: Dựa vào 4 góc đã xác định, chúng ta có thể áp dụng phép biến đổi hình học ngược (perspective transform) để "làm phẳng" biển số, chuyển từ hình thang, hình bốn cạnh bất kỳ về hình chữ nhật chuẩn.

    # 	6. Trích xuất biển số: Cắt vùng chứa biển số từ ảnh gốc và xử lý tiếp.


class Predictor:

    # Đây là một ma trận 3x4 biểu diễn tọa độ chuẩn hóa của bốn góc của một hình chữ nhật trong không gian đồng nhất (homogeneous coordinates).

	# * Hàng đầu tiên [-.5, .5, .5, -.5] biểu diễn tọa độ x của bốn góc
	# * Hàng thứ hai [-.5, -.5, .5, .5] biểu diễn tọa độ y của bốn góc
	# * Hàng thứ ba [1., 1., 1., 1.] là hệ số đồng nhất


    # Đây là hằng số tỷ lệ được sử dụng trong quá trình xử lý hình ảnh.

    # Trong thực tế, hằng số này quyết định mức độ "zoom" khi cắt biển số xe từ ảnh gốc. 
    # Giá trị 7.75 đã được tinh chỉnh để đảm bảo rằng khi cắt biển số, vẫn có đủ không gian xung 
    # quanh để không làm mất chi tiết quan trọng và dễ dàng xử lý trong các bước tiếp theo.
   
    #Stride (bước trượt) là tham số quan trọng trong mạng nơ-ron tích chập (CNN). 
    # Giá trị 16 cho biết kích thước bước trượt khi mạng quét qua ảnh đầu vào.
 
    _q = np.array([
        [-.5, .5, .5, -.5],
        [-.5, -.5, .5, .5],
        [1., 1., 1., 1.]
    ])
    _scaling_const = 7.75
    _stride = 16

    # * self.wpodnet.eval(): Chuyển mô hình sang chế độ đánh giá (evaluation mode), không phải chế độ huấn luyện.
    #  Trong chế độ này, các tham số như Dropout và BatchNorm hoạt động khác so với khi huấn luyện, giúp dự đoán ổn định hơn.

    def __init__(self, wpodnet: WPODNet):
        self.wpodnet = wpodnet
        self.wpodnet.eval()

	# * Vai trò: Thay đổi kích thước ảnh theo tỷ lệ cố định và đảm bảo kích thước cuối là bội của _stride
	# * Tham số:

	# 	* self: Tham chiếu đến đối tượng hiện tại
	# 	* image: Đối tượng ảnh PIL (Python Imaging Library)
	# 	* dim_min: Kích thước tối thiểu mong muốn (pixel)
	# 	* dim_max: Kích thước tối đa được phép (pixel)
	# * Giá trị trả về: Ảnh PIL đã được điều chỉnh kích thước

    def _resize_to_fixed_ratio(self, image: Image.Image, dim_min: int, dim_max: int) -> Image.Image:
        h, w = image.height, image.width

        wh_ratio = max(h, w) / min(h, w)
        side = int(wh_ratio * dim_min)
        bound_dim = min(side + side % self._stride, dim_max)

        factor = bound_dim / max(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)

        # Ensure the both width and height are the multiply of `self._stride`
        reg_w_mod = reg_w % self._stride
        if reg_w_mod > 0:
            reg_w += self._stride - reg_w_mod

        reg_h_mod = reg_h % self._stride
        if reg_h_mod > 0:
            reg_h += self._stride - reg_h % self._stride

        return image.resize((reg_w, reg_h))


    # Dòng code này có mục đích chuyển đổi một tensor hình ảnh từ định dạng (C, H, W) sang (1, C, H, W) bằng cách thêm một chiều batch vào đầu tensor. Cụ thể:

    # 	* Khi một hình ảnh được chuyển đổi thành tensor bằng hàm to_tensor(), nó có kích thước (C, H, W) - đại diện cho số kênh màu (C), chiều cao (H) và chiều rộng (W)
    # 	* Mạng neural thường yêu cầu đầu vào có định dạng batch, tức là (B, C, H, W), trong đó B là kích thước batch
    # 	* Thêm chiều tại vị trí 0 chuyển tensor từ (C, H, W) thành (1, C, H, W), tức là một batch chứa một hình ảnh duy nhất

    def _to_torch_image(self, image: Image.Image) -> torch.Tensor:
        tensor = to_tensor(image)
        return tensor.unsqueeze_(0)

    #     * Ý nghĩa: Đây là một phương thức của lớp (thể hiện qua tham số self), nhận vào một tensor hình ảnh và trả về một cặp mảng NumPy.
    #     * Tham số:

    #         * self: Tham chiếu đến đối tượng hiện tại.
    #         * image: Một tensor PyTorch chứa hình ảnh đầu vào, đã được chuẩn bị sẵn sàng cho mạng neural.
    #     * Kiểu trả về: Một tuple gồm hai mảng NumPy - ma trận xác suất và ma trận biến đổi affine.

    # Ví dụ thực tế: Giả sử bạn có một hình ảnh xe hơi, đã được chuyển thành tensor PyTorch kích thước (1, 3, 224, 224) - tức là 1 hình ảnh, 3 kênh màu RGB, kích thước 224x224 pixel.

    def _inference(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        
        # 	* Ý nghĩa:

		# * with torch.no_grad(): Đây là một context manager trong PyTorch, báo hiệu rằng các hoạt động bên trong không cần tính toán gradient. Điều này giúp giảm bộ nhớ sử dụng và tăng tốc độ khi chỉ thực hiện inference (không phải training).
		# * self.wpodnet.forward(image): Gọi phương thức forward của mô hình WPODNet (được lưu trong thuộc tính wpodnet của đối tượng), truyền vào tensor hình ảnh, và nhận về hai tensor: xác suất và biến đổi affine.

        with torch.no_grad():
            probs, affines = self.wpodnet.forward(image)

        # Convert to squeezed numpy array
        # grid_w: The number of anchors in row
        # grid_h: The number of anchors in column
        probs = np.squeeze(probs.cpu().numpy())[0]     # (grid_h, grid_w)

        # * Ý nghĩa:

        # 	* probs.cpu(): Đưa tensor từ GPU (nếu có) về CPU.
        # 	* .numpy(): Chuyển đổi từ tensor PyTorch sang mảng NumPy.
        # 	* np.squeeze(): Loại bỏ các chiều có kích thước là 1.
        # 	* [0]: Lấy phần tử đầu tiên theo chiều batch.
        # 	* Kết quả là một mảng 2 chiều có kích thước (grid_h, grid_w), trong đó mỗi phần tử biểu thị xác suất xuất hiện biển số tại vị trí đó trên lưới mốc (anchor grid).
        probs = np.squeeze(probs.cpu().numpy())[0]     # (grid_h, grid_w)
        affines = np.squeeze(affines.cpu().numpy())  # (6, grid_h, grid_w)

        return probs, affines

    #         Giả sử probs là một mảng 4x4 như sau:

    # [
    #     [0.12, 0.04, 0.01, 0.09],
    #     [0.05, 0.03, 0.07, 0.02],
    #     [0.06, 0.83, 0.15, 0.22], ← Hàng chứa giá trị lớn nhất
    #     [0.11, 0.13, 0.25, 0.14]
    # ]
    # Lần lượt thực hiện các bước:

    # probs.argmax():

    # Tìm giá trị lớn nhất trong mảng, đó là 0.83 ở vị trí (2,1)
    # Trong bộ nhớ, NumPy lưu trữ mảng theo thứ tự phẳng (flatted order), nên vị trí (2,1) tương ứng với chỉ số phẳng: 2*4 + 1 = 9
    # probs.argmax() trả về 9
    # probs.shape:

    # Trả về (4, 4) - kích thước của mảng 4 hàng, 4 cột
    # np.unravel_index(9, (4, 4)):

    # Chuyển đổi chỉ số phẳng 9 thành tọa độ 2D trong mảng kích thước (4,4)
    # Tính toán: 9 = 2*4 + 1, nên tọa độ là (2,1)
    # Trả về tuple (2, 1)
    # Kết quả cuối cùng: return (2, 1)

    def _get_max_anchor(self, probs: np.ndarray) -> Tuple[int, int]:
        return np.unravel_index(probs.argmax(), probs.shape)

	# * Chức năng: Hàm này tính toán các điểm biên của biển số xe dựa trên ma trận affine và vị trí anchor (điểm neo).
	# * Tham số:

	# 	* self: Tham chiếu đến đối tượng hiện tại của lớp Predictor
	# 	* affines: Ma trận affine dạng numpy array, thông thường có kích thước (6, H, W) với H và W là chiều cao và chiều rộng của lưới đặc trưng
	# 	* anchor_y: Tọa độ y của điểm neo (điểm có xác suất phát hiện biển số cao nhất)
	# 	* anchor_x: Tọa độ x của điểm neo
	# 	* scaling_ratio: Tỷ lệ tỷ lệ co giãn, mặc định là 1.0
	# * Giá trị trả về: Một mảng numpy chứa tọa độ của các đỉnh của biển số xe đã được chuẩn hóa

    def _get_bounds(self, affines: np.ndarray, anchor_y: int, anchor_x: int, scaling_ratio: float = 1.0) -> np.ndarray:
        # Compute theta
        theta = affines[:, anchor_y, anchor_x]
        theta = theta.reshape((2, 3))
        theta[0, 0] = max(theta[0, 0], 0.0)
        theta[1, 1] = max(theta[1, 1], 0.0)

        # Convert theta into the bounding polygon
        bounds = np.matmul(theta, self._q) * self._scaling_const * scaling_ratio

        # Normalize the bounds
        _, grid_h, grid_w = affines.shape
        bounds[0] = (bounds[0] + anchor_x + .5) / grid_w
        bounds[1] = (bounds[1] + anchor_y + .5) / grid_h

        return np.transpose(bounds)

    # 	Phân tích chi tiết hàm predict trong WPODNet-Pytorch
    # Hàm predict này là một phương thức của class Predictor trong mô hình WPODNet dùng để phát hiện biển số xe trong ảnh. Tôi sẽ phân tích chi tiết từng dòng code:

    # 1. Khai báo hàm và tham số
    # def predict(self, image: Image.Image, scaling_ratio: float = 1.0, dim_min: int = 288, dim_max: int = 608) -> Prediction:

    # 	* Ý nghĩa: Đây là method phát hiện biển số xe từ ảnh đầu vào.
    # 	* Tham số:

    # 		* self: Tham chiếu đến instance của class Predictor
    # 		* image: Ảnh đầu vào dạng đối tượng Image của thư viện PIL
    # 		* scaling_ratio: Tỷ lệ co giãn, mặc định là 1.0
    # 		* dim_min: Kích thước nhỏ nhất của ảnh sau khi resize, mặc định là 288 pixel
    # 		* dim_max: Kích thước lớn nhất của ảnh sau khi resize, mặc định là 608 pixel
    # 	* Kiểu trả về: Đối tượng Prediction chứa kết quả phát hiện biển số xe

    # Ví dụ thực tế: Khi người dùng muốn phát hiện biển số từ ảnh chụp trên đường:

    # predictor = Predictor()
    # image = Image.open("xe_tren_duong.jpg")
    # result = predictor.predict(image)

    # 2. Lấy kích thước ảnh gốc
    # orig_h, orig_w = image.height, image.width

    # 	* Ý nghĩa: Lưu lại kích thước ban đầu của ảnh (chiều cao và chiều rộng) để sử dụng sau này khi chuyển đổi tọa độ từ ảnh đã resize về ảnh gốc.
    # 	* Chi tiết: 

    # 		* orig_h: chiều cao của ảnh gốc tính theo pixel
    # 		* orig_w: chiều rộng của ảnh gốc tính theo pixel

    # Ví dụ thực tế: Nếu ảnh đầu vào có kích thước 1920x1080 pixels:

    # # Khi đó orig_h = 1080, orig_w = 1920

    # 3. Resize ảnh để chuẩn bị xử lý
    # # Resize the image to fixed ratio
    # # This operation is convienence for setup the anchors
    # resized = self._resize_to_fixed_ratio(image, dim_min=dim_min, dim_max=dim_max)

    # 	* Ý nghĩa: Thay đổi kích thước ảnh đầu vào theo tỷ lệ cố định để đảm bảo kích thước phù hợp với mô hình WPODNet.
    # 	* Chi tiết: Phương thức _resize_to_fixed_ratio thực hiện:

    # 		* Giữ tỷ lệ khung hình của ảnh gốc
    # 		* Đảm bảo chiều dài nhỏ nhất không dưới dim_min (288 pixel)
    # 		* Đảm bảo chiều dài lớn nhất không vượt quá dim_max (608 pixel)

    # Ví dụ thực tế: Nếu ảnh đầu vào có kích thước 1920x1080:

    # # Ảnh sẽ được resize thành khoảng 608x342 pixels
    # # Vì chiều rộng 1920 > dim_max (608), nên sau khi resize tỷ lệ:
    # # Chiều rộng mới = 608
    # # Chiều cao mới = 1080 * (608/1920) ≈ 342

    # 4. Chuyển đổi định dạng ảnh để phù hợp với mô hình
    # resized = self._to_torch_image(resized)

    # 	* Ý nghĩa: Chuyển đổi ảnh từ định dạng PIL.Image sang định dạng tensor của PyTorch để mô hình có thể xử lý.
    # 	* Chi tiết: Phương thức _to_torch_image thực hiện:

    # 		* Chuyển ảnh từ định dạng RGB sang tensor
    # 		* Chuẩn hóa giá trị pixel
    # 		* Thay đổi thứ tự kênh màu từ HWC (Height, Width, Channel) sang CHW (Channel, Height, Width)

    # Ví dụ thực tế:

    # # Ảnh RGB kích thước 608x342 sẽ được chuyển thành tensor kích thước (3, 342, 608)
    # # Trong đó 3 tương ứng với 3 kênh màu RGB, giá trị pixel được chuẩn hóa về khoảng [0,1]

    # 5. Chuyển tensor sang thiết bị (CPU/GPU) phù hợp
    # resized = resized.to(self.wpodnet.device)

    # 	* Ý nghĩa: Chuyển tensor ảnh sang thiết bị (CPU/GPU) mà mô hình WPODNet đang chạy.
    # 	* Chi tiết: 

    # 		* Nếu mô hình đang chạy trên GPU, tensor sẽ được chuyển sang GPU để tăng tốc quá trình tính toán
    # 		* Nếu mô hình chạy trên CPU, tensor vẫn sẽ nằm ở CPU

    # Ví dụ thực tế:

    # # Nếu mô hình đang chạy trên GPU NVIDIA
    # # Tensor sẽ được chuyển từ CPU sang GPU để tính toán nhanh hơn khoảng 10-50 lần

    # 6. Thực hiện suy luận (inference) với mô hình WPODNet
    # # Inference with WPODNet
    # # probs: The probability distribution of the location of license plate
    # # affines: The predicted affine matrix
    # probs, affines = self._inference(resized)

    # 	* Ý nghĩa: Đưa ảnh đã xử lý qua mô hình WPODNet để nhận kết quả dự đoán.
    # 	* Chi tiết: Phương thức _inference thực hiện:

    # 		* Đưa tensor ảnh qua mạng neural WPODNet
    # 		* Trả về hai kết quả chính:

    # 			* probs: Ma trận xác suất biểu thị khả năng xuất hiện biển số tại mỗi vị trí
    # 			* affines: Ma trận chứa các tham số biến đổi affine dự đoán vị trí và hình dạng của biển số

    # Ví dụ thực tế:

    # # probs là ma trận (ví dụ 24x38) chứa xác suất ở mỗi vị trí grid
    # # affines là ma trận (ví dụ 6x24x38) chứa thông tin biến đổi affine ở mỗi vị trí
    # # Mỗi vị trí có 6 tham số affine để xác định hình dạng, kích thước và góc nghiêng của biển số

    # 7. Lấy xác suất cao nhất từ ma trận xác suất
    # # Get the theta with maximum probability
    # max_prob = np.amax(probs)

    # 	* Ý nghĩa: Tìm giá trị xác suất cao nhất trong ma trận xác suất probs.
    # 	* Chi tiết: 

    # 		* np.amax() là hàm của thư viện NumPy để tìm giá trị lớn nhất trong mảng
    # 		* Giá trị này thể hiện mức độ tin cậy cao nhất về việc tìm thấy biển số

    # Ví dụ thực tế:

    # # Nếu probs là ma trận xác suất với giá trị từ 0 đến 1
    # # Giả sử điểm có xác suất cao nhất là 0.92
    # # Khi đó max_prob = 0.92, nghĩa là mô hình có 92% độ tin cậy về vị trí biển số

    # 8. Xác định vị trí có xác suất cao nhất
    # anchor_y, anchor_x = self._get_max_anchor(probs)

    # 	* Ý nghĩa: Tìm tọa độ (y, x) của điểm có xác suất cao nhất trong ma trận probs.
    # 	* Chi tiết: Phương thức _get_max_anchor trả về tọa độ hàng (y) và cột (x) của phần tử có giá trị lớn nhất trong ma trận xác suất.

    # Ví dụ thực tế:

    # # Nếu probs là ma trận 24x38 và điểm có xác suất cao nhất nằm ở vị trí hàng 15, cột 22
    # # anchor_y = 15, anchor_x = 22
    # # Đây là vị trí trung tâm của biển số được phát hiện trong không gian lưới của mô hình

    # 9. Tính toán biên của biển số xe từ ma trận affine
    # bounds = self._get_bounds(affines, anchor_y, anchor_x, scaling_ratio)

    # 	* Ý nghĩa: Tính toán tọa độ bốn góc của biển số xe từ ma trận biến đổi affine tại vị trí có xác suất cao nhất.
    # 	* Chi tiết: Phương thức _get_bounds:

    # 		* Lấy các tham số affine tại vị trí (anchor_y, anchor_x)
    # 		* Áp dụng biến đổi affine để tính toán bốn góc của biển số
    # 		* Áp dụng tỷ lệ co giãn (scaling_ratio) nếu cần

    # Ví dụ thực tế:

    # # Giả sử ảnh đã resize có kích thước 608x342
    # # Ma trận affine tại vị trí (15, 22) chứa thông tin về hình dạng và hướng của biển số
    # # bounds là ma trận 4x2 chứa tọa độ 4 góc của biển số trong không gian chuẩn hóa [0,1]
    # # Ví dụ: [(0.3, 0.4), (0.5, 0.4), (0.5, 0.5), (0.3, 0.5)]

    # 10. Chuyển đổi tọa độ về ảnh gốc
    # bounds[:, 0] *= orig_w
    # bounds[:, 1] *= orig_h

    # 	* Ý nghĩa: Chuyển đổi tọa độ các góc của biển số về hệ tọa độ của ảnh gốc.
    # 	* Chi tiết:

    # 		* bounds[:, 0]: lấy tất cả các tọa độ x (cột 0) của ma trận bounds
    # 		* bounds[:, 1]: lấy tất cả các tọa độ y (cột 1) của ma trận bounds
    # 		* Nhân với kích thước gốc để chuyển từ không gian chuẩn hóa [0,1] về không gian pixel thực tế

    # Ví dụ thực tế:

    # # Nếu ảnh gốc có kích thước 1920x1080
    # # Các tọa độ chuẩn hóa: [(0.3, 0.4), (0.5, 0.4), (0.5, 0.5), (0.3, 0.5)]
    # # Sau khi chuyển đổi: [(576, 432), (960, 432), (960, 540), (576, 540)]
    # # Đây là tọa độ pixel thực tế của bốn góc biển số trên ảnh gốc

    # 11. Tạo và trả về đối tượng Prediction chứa kết quả
    # return Prediction(
    #     image=image,
    #     bounds=bounds.astype(np.int32),
    #     confidence=max_prob.item()
    # )

    # 	* Ý nghĩa: Đóng gói kết quả phát hiện vào đối tượng Prediction và trả về.
    # 	* Chi tiết:

    # 		* image: ảnh gốc đầu vào
    # 		* bounds: tọa độ của bốn góc biển số, chuyển về kiểu số nguyên (int32)
    # 		* confidence: độ tin cậy của kết quả (xác suất cao nhất), chuyển từ tensor sang số thực

    # Ví dụ thực tế:

    # # Kết quả trả về là đối tượng Prediction với:
    # # - Ảnh gốc: xe_tren_duong.jpg
    # # - Tọa độ biển số: [(576, 432), (960, 432), (960, 540), (576, 540)]
    # # - Độ tin cậy: 0.92 (92%)

    # Áp dụng thực tế:
    # Trong một hệ thống nhận dạng biển số xe tự động tại bãi đỗ xe:

    # 	1. Hệ thống chụp ảnh xe khi xe tiến vào cổng
    # 	2. Ảnh được đưa vào hàm predict của mô hình WPODNet
    # 	3. Hàm trả về vị trí chính xác của biển số xe trong khung hình
    # 	4. Phần biển số được cắt ra và đưa vào mô hình OCR để nhận dạng ký tự
    # 	5. Biển số được so sánh với cơ sở dữ liệu để xác định xe có quyền vào bãi không
    # 	6. Barrier tự động mở nếu xe được phép vào

    def predict(self, image: Image.Image, scaling_ratio: float = 1.0, dim_min: int = 288, dim_max: int = 608) -> Prediction:
        orig_h, orig_w = image.height, image.width

        # Resize the image to fixed ratio
        # This operation is convienence for setup the anchors
        resized = self._resize_to_fixed_ratio(image, dim_min=dim_min, dim_max=dim_max)
        resized = self._to_torch_image(resized)
        resized = resized.to(self.wpodnet.device)

        # Inference with WPODNet
        # probs: The probability distribution of the location of license plate
        # affines: The predicted affine matrix
        probs, affines = self._inference(resized)

        # Get the theta with maximum probability
        max_prob = np.amax(probs)
        anchor_y, anchor_x = self._get_max_anchor(probs)
        bounds = self._get_bounds(affines, anchor_y, anchor_x, scaling_ratio)

        bounds[:, 0] *= orig_w
        bounds[:, 1] *= orig_h

        return Prediction(
            image=image,
            bounds=bounds.astype(np.int32),
            confidence=max_prob.item()
        )
