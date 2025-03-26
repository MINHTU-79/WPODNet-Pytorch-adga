import torch
import torch.nn as nn


	# * Tham số đầu vào:


	# 	* in_channels: Số kênh đầu vào (độ sâu của tensor đầu vào)
	# 	* out_channels: Số kênh đầu ra (số lượng bộ lọc tích chập)
	# * Thành phần cấu tạo:


	# 	a. self.conv_layer: Lớp tích chập 2D với kernel size 3x3 và padding 1
	# 	b. self.bn_layer: Lớp chuẩn hóa batch với các tham số momentum và epsilon
	# 	c. self.act_layer: Hàm kích hoạt ReLU




class BasicConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(BasicConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_layer = nn.BatchNorm2d(out_channels, momentum=0.99, eps=0.001)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        return self.act_layer(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv_block = BasicConvBlock(channels, channels)
        self.sec_layer = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_layer = nn.BatchNorm2d(channels, momentum=0.99, eps=0.001)
        self.act_layer = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.sec_layer(h)
        h = self.bn_layer(h)
        return self.act_layer(x + h)

# self.backbone là thành phần chính của mạng, được xây dựng bằng cách sử dụng nn.Sequential, 
# cho phép nhiều lớp mạng nơ-ron được xếp chồng lên nhau theo thứ tự. Ảnh đầu vào sẽ đi qua backbone 
# để trích xuất các đặc trưng.

class WPODNet(nn.Module):
    def __init__(self):
        super(WPODNet, self).__init__()
        self.backbone = nn.Sequential(
            BasicConvBlock(3, 16),
            BasicConvBlock(16, 16),
            nn.MaxPool2d(2),
            BasicConvBlock(16, 32),
            ResBlock(32),
            nn.MaxPool2d(2),
            BasicConvBlock(32, 64),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            BasicConvBlock(64, 64),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2),
            BasicConvBlock(64, 128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        self.prob_layer = nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.bbox_layer = nn.Conv2d(128, 6, kernel_size=3, padding=1)


        # Đoạn mã này tạo một tensor giả "dummy" chỉ để theo dõi thiết bị mà mô hình đang chạy trên đó (CPU hoặc GPU).
        # Phương thức device là một property cho phép truy cập dễ dàng đến thông tin thiết bị mà không cần lưu trữ thiết bị một cách rõ ràng.

        # Ví dụ thực tế: Giống như cách một ứng dụng điện thoại thông minh tự động xác định liệu nó đang chạy trên iPhone hay Android 
        # để tối ưu hóa hiệu suất, tensor dummy giúp mô hình biết nó đang chạy trên CPU hay GPU để xử lý dữ liệu một cách phù hợp.
        
        # Registry a dummy tensor for retrieve the attached device
        self.register_buffer('dummy', torch.Tensor(), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.dummy.device

    def forward(self, image: torch.Tensor):
        feature: torch.Tensor = self.backbone(image)
        probs: torch.Tensor = self.prob_layer(feature)
        probs = torch.softmax(probs, dim=1)
        affines: torch.Tensor = self.bbox_layer(feature)

        return probs, affines
