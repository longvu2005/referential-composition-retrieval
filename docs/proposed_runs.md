# Chạy proposed bằng script

Bản bổ sung dựa trên repo tại commit `022925c73c4bd81790ddb93ca3843ddf829aee2d`
và tiếp nối bản sửa baseline trước đó. Giải nén các file vào đúng thư mục gốc repo.
File YAML trong ZIP dùng đường dẫn mặc định; nếu đã đặt đường dẫn riêng, điền lại
hoặc dùng các cờ override bên dưới. ZIP không chứa checkpoint, dataset hoặc cache.

## Các lệnh chính

Cài môi trường `.venv-proposed` theo phần Setup trong README, cấu hình đường dẫn
dữ liệu rồi chạy từ thư mục gốc repo:

```bash
# 1. Tạo cache: dùng configs/methods/proposed/build_cache.yaml.
bash scripts/build_cache.bash

# 2. Train từ cache có sẵn, chọn best.pt trên val, đánh giá cả val và test.
bash scripts/run_proposed.bash --train

# 3. Khi đã có cache + best.pt + tokenizer/: chỉ retrieve và evaluate.
bash scripts/run_proposed.bash

# Hoặc từ đầu: build cache -> train -> retrieve/evaluate val -> test.
bash scripts/run_proposed.bash --build-cache --train
```

`--train` là huấn luyện mới, không phải resume. Để tạo thí nghiệm riêng mà không
ghi đè checkpoint hiện có, dùng `--output-dir runs/proposed_exp2`.
Không có `--build-cache` thì runner không chạy lại detector/DINOv3. Cache tạo bởi
pipeline cũ vẫn dùng được nếu đầy đủ và đúng dataset; thay đổi này không bắt buộc
build lại cache. Lệnh build cache riêng sẽ thực hiện build, không tự bỏ qua hay
tiếp tục từ giữa một lần build bị ngắt.

Một cache mới có `cache_id` mới. Vì checkpoint được gắn với cache khi train,
`--build-cache` trong runner yêu cầu đi kèm `--train`. Nếu chỉ muốn build rồi
train vào lúc khác, dùng script `build_cache.bash` riêng.

## Chọn split, checkpoint và môi trường

```bash
bash scripts/run_proposed.bash --splits val
bash scripts/run_proposed.bash --splits test

# Checkpoint khác; giữ tokenizer/ cùng cấp với file checkpoint.
bash scripts/run_proposed.bash \
  --checkpoint runs/proposed_exp2/best.pt \
  --output-dir runs/proposed_exp2_eval

# Dùng Python khác đã cài dependencies.
PROPOSED_PYTHON=python bash scripts/run_proposed.bash --splits test
```

Mặc định đánh giá cả val/test bằng **cùng một checkpoint**. Khi có `--train`,
runner dùng `best.pt` vừa tạo từ output training thực tế; không đọc nhầm checkpoint
cũ trong retrieve YAML. Việc chọn best vẫn dựa trên Full-mAP của toàn bộ val.
`--splits` chỉ chọn split đánh giá cuối cùng, không đổi split chọn checkpoint.
`--checkpoint` dành cho evaluate-only và không kết hợp với `--train`.

## Config và đường dẫn trên Kaggle

Các config riêng vẫn giữ vai trò hiện tại:

| Tham số runner | File mặc định |
|---|---|
| `--config` | `configs/methods/proposed/retrieve.yaml` |
| `--train-config` | `configs/methods/proposed/train.yaml` |
| `--cache-config` | `configs/methods/proposed/build_cache.yaml` |

`build_cache.bash --config path/to/build.yaml` chuyển config trực tiếp cho bước build.
Runner dùng chính retrieve YAML cho cả retrieval và evaluator chung; `candidate_ks`
nằm trong retrieve YAML. `evaluate.yaml` vẫn phục vụ lệnh evaluate riêng cũ.
Các đường dẫn tương đối của runner được tính từ thư mục gốc repo.

Có thể ghi đè đồng bộ đường dẫn cho mọi bước đang chạy:

```bash
bash scripts/run_proposed.bash --build-cache --train \
  --final-dir dataset/data/final \
  --image-root /kaggle/input/your-pipa-images \
  --cache /kaggle/working/rcr-cache \
  --output-dir /kaggle/working/proposed

# Dùng cache đã lưu làm Kaggle Dataset: không build lại.
bash scripts/run_proposed.bash --train \
  --cache /kaggle/input/your-cache-dataset/cache \
  --output-dir /kaggle/working/proposed
```

Thay các đường dẫn ví dụ bằng đường dẫn thực tế. Trong notebook, đặt `%%bash`
ở đầu cell và `cd /kaggle/working/referential-composition-retrieval` trước các lệnh.
`--output-dir` đặt thư mục kết quả và, nếu train, cả thư mục checkpoint/tokenizer.
Ở chế độ evaluate-only, cờ này không thay checkpoint; chọn model bằng `--checkpoint`.
W&B giữ nguyên cấu hình trong train YAML. Muốn chạy không cần đăng nhập W&B,
đặt `wandb.enabled: false` hoặc chọn chế độ offline phù hợp trong config.

## Các điểm kiểm tra và output

- Các bước đang chạy phải dùng cùng `final_dir`, `image_root`, `cache`; phát hiện
  lệch config trước khi chạy stage. Các cờ override áp dụng cho mọi stage.
- `evaluation.enabled` phải bật khi train qua runner. `top_m` của validation khi
  training phải trùng với retrieval cuối, để nhất quán với cách chọn checkpoint.
- Thiếu cache, cache đang build dở, thiếu checkpoint/tokenizer hoặc stage thất bại
  thì dừng; không tiếp tục evaluate bằng kết quả còn sót của lần trước.
- Retrieval kiểm tra gallery và `cache_id` trước khi tải text backbone.
- Mỗi stage chạy bằng cùng Python trong process riêng, giải phóng bộ nhớ khi xong.
- Retrieval dùng cách lưu ranking/metadata chung với baseline và xóa metrics cũ.
  Model, loss, thuật toán build cache và công thức metric được giữ nguyên.

Trong `runs/proposed` hoặc `--output-dir`:

| Đường dẫn | Nội dung |
|---|---|
| `best.pt`, `last.pt`, `tokenizer/` | Do training tạo; best được chọn trên val |
| `runner_configs/` | Config đã áp dụng override cho từng stage |
| `val/`, `test/` | `rankings.pt`, `run.json`, `metrics.json` |
| `summary.csv` | Metric của các split trong lần gọi runner hiện tại |

Summary chỉ được ghi sau khi mọi stage được yêu cầu thành công. Lần chạy test-only
tạo summary chứa test; metrics của val từ lần trước vẫn nằm trong `val/`.
Proposed không sinh `scores.npy` toàn gallery vì dùng coarse retrieval và fine
reranking; `rankings.pt` có thêm `coarse_topm` cho CandidateRecall.

## Lệnh thấp hơn vẫn dùng được

```bash
.venv-proposed/bin/python tools/methods/retrieve_proposed.py --split val
.venv-proposed/bin/python tools/methods/evaluate_proposed.py --split val

# Cùng evaluator với baseline, dùng luôn retrieve config.
.venv-proposed/bin/python tools/methods/evaluate.py \
  --config configs/methods/proposed/retrieve.yaml --split val
```

Config mặc định dùng `{split}` trong đường dẫn output nên val/test không ghi đè
nhau. Với config cũ tự sửa đang hard-code `test`, đổi thành `{split}` hoặc truyền
`--output-dir` cho runner.

## Kiểm chứng bản sửa

Toàn bộ test suite: **298 passed, 1 skipped**; test bỏ qua yêu cầu CUDA để kiểm tra
VRAM. Ruff, kiểm tra cú pháp Bash và `git diff --check` đạt.

Test tích hợp thực thi các hàm build cache, train, retrieval và evaluator với
dữ liệu nhỏ cùng detector/image/text encoder nhỏ. Có kiểm tra cùng checkpoint cho
val/test, không rebuild/retrain ở lần evaluate sau, config override, dừng khi stage
lỗi và từ chối checkpoint dùng cache khác trước khi tải backbone. Shell wrapper
được kiểm tra riêng cho môi trường Python, working directory và đối số có dấu cách.

Chưa chạy pretrained DINOv3/Grounding DINO/BERT trên dữ liệu thật hoặc GPU trong
môi trường kiểm thử này; kết quả test trên xác nhận logic và tích hợp của code.
