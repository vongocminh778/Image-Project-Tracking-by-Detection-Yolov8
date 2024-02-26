# Image-Project-Tracking-by-Detection-Yolov8

1. Thiết lập môi trường Conda
Đầu tiên, hãy tạo một môi trường Conda mới. Mở terminal của bạn và chạy lệnh sau:

```
conda create --name ultralytics-env python=3.8 -y
```
2. Kích hoạt môi trường mới:

```
conda activate ultralytics-env
```
3. Cài đặt Ultralytics
Bạn có thể cài đặt Ultralytics Gói từ Conda. Thực hiện lệnh sau:
```
conda install -c conda-forge ultralytics
```
4. Download MOT17 dataset
- Cách 1:
```
wget https://motchallenge.net/data/MOT17.zip
```
- Cách 2:
```
gdown 1vOj9OpxeyozWzpPCtUY7fDVaBQwsPM9n
```

<video width="640" height="480" controls>
  <source src="./notebook/tracking_results/result_compressed.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
