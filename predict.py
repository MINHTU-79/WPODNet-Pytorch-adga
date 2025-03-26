    # Module errno trong Python chứa các mã lỗi hệ thống tiêu chuẩn. 
    # Khi một thao tác hệ thống (như mở tệp, truy cập mạng) thất bại,
    #  Python thường ném ra một ngoại lệ OSError hoặc một lớp con của nó, 
    # kèm theo mã lỗi. Module errno chứa các hằng số đại diện cho các mã lỗi này,
    #  giúp bạn xác định chính xác loại lỗi gặp phải.
import errno

	# 1. Module argparse: Đây là một module tiêu chuẩn của Python được thiết kế để tạo giao diện dòng lệnh (CLI) thân thiện với người dùng. Nó tự động tạo trợ giúp và thông báo lỗi, xử lý các đối số dòng lệnh.

	# 2. ArgumentParser: Đây là lớp chính trong module argparse, dùng để tạo một parser xử lý các đối số dòng lệnh. Nó cho phép:


	# 	* Định nghĩa các tham số dòng lệnh (bắt buộc/tùy chọn)
	# 	* Gán kiểu dữ liệu cho tham số
	# 	* Thiết lập giá trị mặc định
	# 	* Thêm mô tả trợ giúp
	# 	* Tự động tạo trợ giúp (--help)
	# 3. ArgumentTypeError: Đây là loại lỗi được ném ra khi chuyển đổi một đối số thành một kiểu cụ thể thất bại. Thường được sử dụng khi bạn viết các hàm xác thực tùy chỉnh cho đối số.
    
from argparse import ArgumentParser, ArgumentTypeError
from pathlib import Path

import torch

from wpodnet.backend import Predictor
from wpodnet.model import WPODNet
from wpodnet.stream import ImageStreamer

# Ý nghĩa
# Đây là một câu lệnh điều kiện kiểm tra xem file Python hiện tại có đang được chạy trực tiếp hay không:

# 	1. Khi file được chạy trực tiếp (như python predict.py): biến đặc biệt __name__ sẽ có giá trị là chuỗi '__main__', khiến điều kiện trở thành True, và code bên trong khối lệnh sẽ được thực thi.

# 	2. Khi file được import vào một file khác (như import predict): biến __name__ sẽ có giá trị là tên module (ví dụ: 'predict'), khiến điều kiện trở thành False, và code bên trong khối lệnh sẽ không được thực thi.


# Mục đích
# 	* Giúp phân biệt khi nào file được chạy trực tiếp và khi nào nó được import để sử dụng như một module
# 	* Cho phép viết code có thể vừa sử dụng như thư viện, vừa có thể chạy độc lập
# 	* Là một mẫu thiết kế phổ biến trong Python để điều khiển luồng thực thi

if __name__ == '__main__':
    parser = ArgumentParser()

    #     Phân tích:

    # add_argument() là phương thức để thêm một tham số mới.
    # 'source' là tên của tham số, không có dấu gạch ngang phía trước, nên đây là tham số vị trí (positional argument).
    # type=str chỉ định rằng giá trị nhập vào phải là chuỗi.
    # help='the path to the image' là thông báo mô tả sẽ hiển thị khi người dùng gọi chương trình với cờ --help.
    # Tham số vị trí là bắt buộc và phải được cung cấp theo đúng thứ tự.

    parser.add_argument(
        'source',
        type=str,
        help='the path to the image'
    )

    # Phân tích:

	# * -w và --weight là bí danh cho cùng một tham số. -w là dạng rút gọn, --weight là dạng đầy đủ.
	# * type=str quy định giá trị phải là chuỗi.
	# * required=True chỉ định rằng tham số này là bắt buộc, mặc dù nó là tham số tùy chọn (có tiền tố - hoặc --).
	# * help='the path to the model weight' cung cấp mô tả khi hiển thị trợ giúp.
    parser.add_argument(
        '-w', '--weight',
        type=str,
        required=True,
        help='the path to the model weight'
    )

    # Phân tích:

    # 	* --scale là tên tham số tùy chọn.
    # 	* type=float xác định rằng giá trị nhập vào phải là số thực.
    # 	* default=1.0 quy định giá trị mặc định 1.0 nếu người dùng không cung cấp giá trị.
    # 	* help='adjust the scaling ratio. default to 1.0.' là thông tin mô tả.
    # 	* Tham số này không bắt buộc do không có required=True và có giá trị mặc định.

    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='adjust the scaling ratio. default to 1.0.'
    )

    # Phân tích:

	# * --save-annotated là tên tham số tùy chọn dùng dấu gạch ngang giữa các từ.
	# * type=str quy định giá trị là chuỗi.
	# * Không có required=True và default, nên tham số này là không bắt buộc và giá trị mặc định là None.
	# * help='save the annotated image at the given folder' mô tả mục đích của tham số.

    parser.add_argument(
        '--save-annotated',
        type=str,
        help='save the annotated image at the given folder'
    )

    # Phân tích:

    # 	* Tương tự như --save-annotated, tham số --save-warped cũng là không bắt buộc.
    # 	* type=str chỉ ra rằng giá trị là chuỗi.
    # 	* help='save the warped image at the given folder' mô tả tác dụng của tham số.
    # 	* Tham số này cho phép lưu ảnh biển số đã được biến đổi góc nhìn để dễ đọc.

    parser.add_argument(
        '--save-warped',
        type=str,
        help='save the warped image at the given folder'
    )

    # * parse_args() là phương thức để phân tích các tham số dòng lệnh thực tế mà người dùng đã nhập.
	# * Kết quả được lưu trong biến args, là một đối tượng mà thuộc tính của nó tương ứng với các tham số đã định nghĩa.
	# * Phương thức này tự động xử lý hiển thị trợ giúp và báo lỗi nếu người dùng cung cấp tham số không hợp lệ.
    args = parser.parse_args()


	# * Cấu trúc: raise [exception_type]([các tham số])
	# * Chức năng: Dùng để ném (throw) một ngoại lệ (exception) trong Python, làm ngắt luồng thực thi bình thường của chương trình
	# * Vai trò trong đoạn code: Nó báo hiệu rằng có lỗi xảy ra và chương trình không thể tiếp tục với giá trị scale hiện tại
    if args.scale <= 0.0:
        raise ArgumentTypeError(message='scale must be greater than 0.0')

    if args.save_annotated is not None:
        save_annotated = Path(args.save_annotated)
        if not save_annotated.is_dir():
            raise FileNotFoundError(errno.ENOTDIR, 'No such directory', args.save_annotated)
    else:
        save_annotated = None

    if args.save_warped is not None:
        save_warped = Path(args.save_warped)
        if not save_warped.is_dir():
            raise FileNotFoundError(errno.ENOTDIR, 'No such directory', args.save_warped)
    else:
        save_warped = None

    # Prepare for the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = WPODNet()
    model.to(device)

    # torch.load() là gì và dùng để làm gì?
    # torch.load() là một hàm quan trọng trong thư viện PyTorch dùng để đọc và tải các đối tượng đã được lưu trữ trước đó bằng torch.save() từ một tệp vào bộ nhớ. Nói một cách đơn giản, nó cho phép bạn:

    # 	1. Khôi phục lại trạng thái của mô hình, bộ tối ưu hóa, hoặc bất kỳ đối tượng Python nào đã được lưu trước đó
    # 	2. Tải lại các trọng số (weights) của mô hình sau khi huấn luyện
    # 	3. Tiếp tục quá trình huấn luyện từ điểm đã dừng trước đó

    # Cú pháp và Tham số
    # torch.load(f, map_location=None, pickle_module=pickle, **pickle_load_args)

    # Trong đó:

    # 	* f: Đường dẫn đến tệp (string) hoặc đối tượng tệp đã mở (file-like object)
    # 	* map_location: Chỉ định cách ánh xạ các tensor vào thiết bị (ví dụ: CPU hoặc GPU)
    # 	* pickle_module: Module dùng để giải mã (unpickling) dữ liệu (mặc định là Python's pickle)


    # Nội dung checkpoint: Một checkpoint có thể chứa:

    # 		* Trọng số của mô hình (model weights)
    # 		* Trạng thái của bộ tối ưu hóa (optimizer state)
    # 		* Epoch cuối cùng
    # 		* Giá trị loss cuối cùng
    # 		* Các thông số khác (hyperparameters)

    #================

    #load_state_dict() là một phương thức quan trọng của các module trong PyTorch (bao gồm cả mô hình neural network) dùng để nạp các tham số đã lưu trữ vào mô hình hiện tại. Phương thức này nhận một Python dictionary làm đầu vào và cập nhật tất cả các tham số của mô hình dựa trên nội dung của dictionary đó.

    # Cú pháp và tham số
    # model.load_state_dict(state_dict, strict=True)

    # Trong đó:

    # 	* state_dict: Dictionary chứa các cặp khóa-giá trị ánh xạ tên tham số đến các giá trị tham số
    # 	* strict: Tham số boolean

    # 		* Khi strict=True (mặc định): Yêu cầu tất cả các khóa trong state_dict phải khớp chính xác với các tham số của mô hình
    # 		* Khi strict=False: Cho phép nạp một phần tham số, bỏ qua các khóa không khớp

    # Cấu trúc chi tiết của state_dict
    # Một state_dict trong PyTorch có cấu trúc như sau:

    # {
    #     "layer1.weight": tensor([...]),
    #     "layer1.bias": tensor([...]),
    #     "layer2.weight": tensor([...]),
    #     "layer2.bias": tensor([...]),
    #     ...
    # }

    # Mỗi khóa trong dictionary này đại diện cho một tham số có thể huấn luyện trong mô hình, thường có định dạng:

    # 	* tên_lớp.weight: Chứa trọng số của lớp
    # 	* tên_lớp.bias: Chứa độ lệch (bias) của lớp
    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint)

    predictor = Predictor(model)

    streamer = ImageStreamer(args.source)
    for i, image in enumerate(streamer):
        prediction = predictor.predict(image, scaling_ratio=args.scale)

        print(f'Prediction #{i}')
        print('  bounds', prediction.bounds.tolist())
        print('  confidence', prediction.confidence)

        if save_annotated:
            annotated_path = save_annotated / Path(image.filename).name
            annotated = prediction.annotate()
            annotated.save(annotated_path)
            print(f'Saved the annotated image at {annotated_path}')

        if save_warped:
            warped_path = save_warped / Path(image.filename).name
            warped = prediction.warp()
            warped.save(warped_path)
            print(f'Saved the warped image at {warped_path}')

        print()
