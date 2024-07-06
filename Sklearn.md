Scikit-learn (viết tắt là sklearn) là một thư viện mã nguồn mở mạnh mẽ và phổ biến trong cộng đồng Python, được sử dụng rộng rãi cho các nhiệm vụ học máy (machine learning). Nó cung cấp một loạt các công cụ hiệu quả và dễ sử dụng cho việc phân tích dữ liệu và xây dựng các mô hình dự đoán.

Trong Scikit-learn, thuật toán K-Means là một trong những phương pháp phổ biến để thực hiện phân cụm (clustering). K-Means được sử dụng để phân chia một tập hợp dữ liệu thành K cụm, trong đó mỗi cụm được xác định bởi trọng tâm (centroid) của nó. Quá trình phân cụm bằng thuật toán K-Means có thể được tóm tắt như sau:

Khởi tạo: Chọn ngẫu nhiên K trọng tâm ban đầu từ tập dữ liệu.
Gán cụm: Gán mỗi điểm dữ liệu vào cụm có trọng tâm gần nhất.
Cập nhật trọng tâm: Tính toán lại trọng tâm của mỗi cụm dựa trên các điểm dữ liệu đã được gán vào cụm đó.
Lặp lại: Lặp lại quá trình gán cụm và cập nhật trọng tâm cho đến khi các trọng tâm không thay đổi hoặc thay đổi rất ít (đạt đến sự hội tụ).
Sử dụng K-Means trong Scikit-learn
Dưới đây là một ví dụ về cách sử dụng thuật toán K-Means với Scikit-learn:
Cài đặt thư viện:

pip install scikit-learn

Import các thư viện cần thiết:

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

Tạo dữ liệu mẫu:

python

# Tạo dữ liệu ngẫu nhiên với 3 cụm
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

Khởi tạo và huấn luyện mô hình K-Means:

# Khởi tạo mô hình KMeans với số cụm là 3
kmeans = KMeans(n_clusters=3, random_state=42)

# Huấn luyện mô hình
kmeans.fit(X)

Dự đoán cụm cho các điểm dữ liệu:

# Dự đoán cụm cho các điểm dữ liệu
y_kmeans = kmeans.predict(X)
Trực quan hóa kết quả:

# Vẽ biểu đồ phân cụm
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Vẽ trọng tâm của các cụm
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')
plt.show()

Các tham số quan trọng của K-Means trong Scikit-learn:
n_clusters: Số lượng cụm mà bạn muốn phân chia.
init: Phương pháp khởi tạo các trọng tâm ban đầu (mặc định là 'k-means++').
max_iter: Số lượng vòng lặp tối đa cho thuật toán K-Means.
random_state: Để đảm bảo tính tái lập của kết quả bằng cách cố định seed cho quá trình ngẫu nhiên.
