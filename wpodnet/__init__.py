__version__ = '1.0.3'

from .backend import Prediction, Predictor

__all__ = [
    'Prediction', 'Predictor'
]

# __all__ = ['Prediction', 'Predictor']
# Đây là danh sách các tên sẽ được import khi người dùng sử dụng câu lệnh from wpodnet import *.

# Chi tiết kỹ thuật:

# 	* __all__ là một biến đặc biệt trong Python xác định danh sách symbols được export
# 	* Khi người dùng thực hiện from module import *, chỉ những tên trong __all__ mới được import
# 	* Giúp kiểm soát namespace và tránh import những biến/hàm/class không mong muốn
# 	* Trong trường hợp này, chỉ hai class Prediction và Predictor sẽ được import

# Cấu trúc:

# 	* Đây là một list chứa các string, mỗi string là tên của một symbol sẽ được export
# 	* List có thể viết trên một dòng hoặc nhiều dòng như trong code của bạn để dễ đọc