# Hướng dẫn bản sửa baseline

Bản sửa dựa trên commit `022925c73c4bd81790ddb93ca3843ddf829aee2d` của
`longvu2005/referential-composition-retrieval`.

## Áp dụng

Giải nén ZIP vào thư mục gốc repo và ghi đè đúng đường dẫn. ZIP chỉ gồm các file
thêm/sửa, không phải một repo đầy đủ. Không có checkpoint hoặc dữ liệu ảnh trong ZIP.
Nếu đã sửa đường dẫn riêng trên Kaggle, điền lại `data.image_root`,
`model.checkpoint`, `cache.dir` trong `configs/methods/baselines/clip.yaml` sau khi
ghi đè. Tham khảo `docs/baselines.md` để cài `.venv-clip` và `.venv-fafa`.

## Chạy ngắn gọn

Tại thư mục gốc repo, sau khi đã cài môi trường và đặt đường dẫn ảnh:

```bash
# Lần đầu: chuẩn bị checkpoint, chạy bốn baseline trên val/test và evaluate.
bash scripts/run_baselines.bash clip --prepare

# Các lần sau: dùng checkpoint/cache sẵn có.
bash scripts/run_baselines.bash clip

# Có thể tách thành hai lần chạy.
bash scripts/run_baselines.bash clip --splits val
bash scripts/run_baselines.bash clip --splits test

# Chọn baseline cụ thể.
bash scripts/run_baselines.bash clip --modes clip_text clip_image
bash scripts/run_baselines.bash clip --modes early_fusion late_fusion

# FAFA: cùng quy trình retrieve + evaluate, dùng môi trường FAFA.
bash scripts/run_baselines.bash fafa --prepare
```

Trên Kaggle, đặt các lệnh trong một cell `%%bash` và `cd` vào repo trước.
Có thể dùng Python đang cài dependencies bằng
`BASELINE_PYTHON=python bash scripts/run_baselines.bash clip`.
Wrapper cũng nhận `--config /duong/dan/config.yaml`; config đó được dùng cho cả
prepare và run. Các script không tự cài package.

## Các thay đổi chính

- Một checkpoint OpenAI CLIP cung cấp cả image encoder và text encoder.
  Bốn baseline CLIP không cần detector, LLM, checkpoint fusion hoặc huấn luyện RCR.
- Có đủ `clip_image`, `clip_text`, `early_fusion`, `late_fusion`; lệnh cũ dùng
  `image`/`text` vẫn được hỗ trợ.
- Cache cả đặc trưng ảnh, text và điểm từng nhánh. Fusion dùng lại các điểm này;
  dò trọng số không chạy encoder hoặc nhân ma trận query-gallery lại mỗi lần.
- Phần cấu hình fusion chỉ cần `image_weight` và `text_weight`. Early fusion
  tương đương cộng đặc trưng rồi chuẩn hóa; late fusion giữ z-score như code gốc.
- Hỗ trợ checkpoint CLIP chính thức dạng TorchScript và raw state dict theo đúng
  kiến trúc OpenAI CLIP. Không coi checkpoint của thư viện khác là tương thích.
- Dùng chung loader, gallery theo split, loại ảnh self, ranking ổn định khi hòa
  điểm và evaluator hiện tại. Luồng proposed và công thức metric không đổi.

## Chọn trọng số

Mặc định thử `image_weight = alpha` từ 0 đến 1, bước 0.05 và
`text_weight = 1 - alpha`. Chọn riêng cho early/late fusion theo `Full-mAP` trên
toàn bộ val; nếu hòa điểm, ưu tiên alpha gần 0.5 nhất rồi alpha nhỏ hơn.
Có thể sửa grid và metric trong `tuning` trước khi thực nghiệm.

Runner lưu lựa chọn trước khi chạy test. `--splits test` bắt buộc có lựa chọn val
phù hợp; không tự dò trọng số trên test và không âm thầm dùng 0.5/0.5.
Nếu checkpoint, text field, precision, grid hoặc dữ liệu val thay đổi, chạy lại
val. Vì fingerprint ảnh gồm đường dẫn/mtime, chuyển ảnh sang nơi khác có thể cần
chạy val lại. Giữ dữ liệu val để kiểm tra lựa chọn khi chạy test riêng.

Đây là trọng số tốt nhất **trên val trong grid đã thử**, không bảo đảm tối ưu trên
test. Test trong snapshot hiện tại chỉ có 14 query; val có 264 query.

## Kết quả

- `runs/clip/<mode>/<split>/scores.npy`: điểm cho toàn bộ gallery.
- `rankings.pt`, `run.json`, `metrics.json` trong cùng thư mục: ranking, cấu hình
  thực dùng và các metric ID/Full cùng kết quả theo case/query.
- `runs/clip/tuning.json`: mọi lần thử trên val, trọng số chọn và provenance.
- `runs/clip/summary.csv`: bảng kết quả của lần gọi runner hiện tại.
- FAFA lưu tương tự dưới `runs/fafa/`; không có bước dò trọng số CLIP cho FAFA.

Một lần chạy val mới thay thế `tuning.json` bằng các fusion mode được yêu cầu.
Nếu muốn đánh giá cả hai fusion trên test, chọn cả hai khi chạy val.
Các lệnh thấp hơn `retrieve_baseline.py`/`evaluate.py` vẫn dùng được; riêng fusion
ở lệnh retrieval thấp hơn dùng trọng số cố định trong YAML. Dùng runner mới để
thực hiện đúng quy trình tự chọn val rồi cố định test.

## Kiểm chứng

- Toàn bộ suite: **285 passed, 1 skipped**. Test bỏ qua cần CUDA để kiểm tra VRAM.
- Ruff, `git diff --check`, cú pháp Bash: đạt.
- Có test val/test cố tình có trọng số tối ưu khác nhau, kiểm tra test dùng đúng
  lựa chọn val; test từ chối lựa chọn cũ; kiểm tra cache và công thức fusion.
- Có kiểm thử subprocess gọi script Bash bằng CLIP native nhỏ với trọng số ngẫu
  nhiên, tokenizer thật, ảnh nhỏ, đủ bốn mode trên val/test và chạy test riêng.
- FAFA adapter/runner được kiểm tra bằng encoder nhỏ và dữ liệu tổng hợp.

Đây là kiểm chứng logic và tích hợp CPU. Chưa chạy CLIP ViT-L/14 pretrained hoặc
FAFA đầy đủ trên ảnh PIPA thật/GPU vì repo không cung cấp ảnh gốc và checkpoint.
Các warning TorchScript phát sinh từ việc trace model nhỏ trong test.
