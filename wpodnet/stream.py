from pathlib import Path
from typing import Generator, Union

from PIL import Image

	# 1. Lớp ImageStreamer được khai báo với phương thức khởi tạo __init__ nhận tham số image_or_folder (có thể là đường dẫn dạng chuỗi hoặc đối tượng Path).

	# 2. Trong phương thức khởi tạo, đường dẫn được chuyển đổi thành đối tượng Path và một generator được tạo bằng phương thức _get_image_generator.

	# 3. Phương thức _get_image_generator xử lý hai trường hợp:


	# 	* Nếu đường dẫn là một tệp, kiểm tra xem có phải tệp ảnh không
	# 	* Nếu đường dẫn là một thư mục, tìm tất cả các tệp ảnh trong thư mục đó và tất cả các thư mục con
	# 	* Nếu không phải cả hai, nó sẽ báo lỗi
	# 4. Phương thức _is_image_file để xác định liệu một tệp có phải là ảnh hợp lệ không bằng cách thử mở và xác minh nó bằng thư viện PIL.

	# 5. Phương thức __iter__ cho phép lớp này hoạt động như một iterator, trả về generator đã tạo.

class ImageStreamer:
    def __init__(self, image_or_folder: Union[str, Path]):
        path = Path(image_or_folder)
        self.generator = self._get_image_generator(path)

# Kiểu dữ liệu Generator có 3 phần
# Khi khai báo kiểu dữ liệu cho generator trong Python, cú pháp là:
# ```
# Generator[KiểuTrảVề, KiểuGửiVào, KiểuKếtThúc]
# ```



# Ý nghĩa của 3 phần:
# KiểuTrảVề: Kiểu dữ liệu mà generator trả về qua yield

# Trong code của bạn: Image.Image (tức là trả về đối tượng ảnh)
# KiểuGửiVào (= None): Generator có thể nhận dữ liệu từ bên ngoài gửi vào

# Khi để là None có nghĩa là "không nhận dữ liệu gửi vào"
# Đây là tính năng ít dùng của generator
# KiểuKếtThúc (= None): Kiểu dữ liệu khi generator kết thúc hoàn toàn

# Khi để là None có nghĩa là "không trả về gì khi kết thúc"

# ===============

# Kiểu dữ liệu Generator có 3 phần
# Khi khai báo kiểu dữ liệu cho generator trong Python, cú pháp là:
# ```
# Generator[KiểuTrảVề, KiểuGửiVào, KiểuKếtThúc]
# ```


    def _get_image_generator(self, path: Path) -> Generator[Image.Image, None, None]:
        if path.is_file():
            image_paths = [path] if self._is_image_file(path) else []
        elif path.is_dir():
            image_paths = [
                p
                for p in path.rglob('**/*')
                if self._is_image_file(p)
            ]
        else:
            raise TypeError(f'Invalid path to images {path}')

        for p in image_paths:
            yield Image.open(p)

    def _is_image_file(self, path: Path) -> bool:
        try:
            image = Image.open(path)
            image.verify()
            return True
        except Exception:
            return False

# Phương thức __iter__ định nghĩa cách một đối tượng trở thành một iterable (có thể lặp qua), cho phép sử dụng đối tượng trong câu lệnh for hoặc bất kỳ ngữ cảnh nào cần một iterable.


    def __iter__(self):
        return self.generator
