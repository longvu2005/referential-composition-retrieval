# Audit dung lượng cache FAFA trên Kaggle

Repo được kiểm tra: `longvu2005/referential-composition-retrieval`,
base commit `ec5a2a2b231545f92f0f8c3b05ffe9a6ad20b0a0` (`FAFA new`).
Ngày audit: 18/10/2026. Gallery trong commit này có **37.107 ảnh**.

## Kết luận từ log

Lần chạy dừng ở `require_disk_space()` trước vòng extraction ảnh:

```text
need about 11.05 GiB free, found 10.44 GiB
```

Đây là thiếu dung lượng đĩa khoảng 0,61 GiB so với mức dự phòng, chưa phải
lỗi hết RAM hoặc VRAM. `cache.storage_dtype` trong config đã là `float16`.
Giảm batch size không làm giảm kích thước cuối của cache.

Hàm kiểm tra giữ dư `max(64 MiB, 10% số byte cần ghi)`. Bản sửa giữ nguyên
quy tắc này và tính lại dung lượng theo cấu trúc lưu trữ đã tối ưu.

## Các phát hiện và thay đổi

| Vị trí | Trước | Sau |
| --- | --- | --- |
| `cache/fafa.py::_build_images` | Cấp phát và lưu `[G,K_max,H]`, bao gồm mọi slot padding của gallery | Lưu `[N,H]` cùng offsets `[G+1]`, chỉ chứa person thực |
| `cache/store.py`, `ranking.py` | Coarse truy cập tensor gallery có padding | Đọc person theo offsets, padding trong từng batch; cache dense cũ vẫn đọc được |
| Global DINO | Đọc lại scene của mọi ảnh và lưu một bản global mean nữa trong index FAFA | Dùng global mean từ DINO; cache legacy thiếu mean thì tính một lần khi cần |
| `cache/native.py::load_targets` | Giữ dictionary chứa tất cả target shard của batch đồng thời với output đã padding | Mỗi ảnh đọc một lần, copy vào các vị trí cần dùng rồi giải phóng shard |
| Đọc shard | Nạp toàn bộ storage trước khi lấy dữ liệu cần dùng | Memory-map shard do repo ghi; tránh cấp phát thêm bản toàn bộ shard khi đọc |
| Dự phòng đĩa khi resume | Trừ kích thước tất cả `.pt`, kể cả file không tái sử dụng được | Chỉ ghi nhận shard đúng cache ID, key, shape, dtype và hữu hạn; giới hạn credit theo kích thước dự kiến |
| `scripts/setup.bash` | Pip có thể giữ wheel tải về bên cạnh package đã cài | Dùng `--no-cache-dir` cho các lần cài mới |

Với `G=37107`, scene width `768`, riêng bản global FP32 trùng lặp là
**108,71 MiB** dữ liệu tensor. Việc này không đồng nghĩa bản sửa giảm RAM/đĩa
toàn pipeline đúng bằng một giá trị cố định: một số cache legacy chưa có global
mean và sẽ cần đọc scene một lần trong lần train/retrieve đầu tiên.

`_build_images` vẫn giữ pooled shard tạm đến khi publish index thành công.
Đây là checkpoint cho resume; nếu dừng giữa chừng thì không mất phần extraction
đã làm. Bản sửa giảm mạnh index cuối, đồng thời giữ cơ chế phục hồi này.

## Tại sao padding toàn gallery gây tốn dung lượng

Đặt:

- `G`: số ảnh; `K_max`: số slot person của index DINO.
- `N = sum(mask)`: tổng person thực.
- `H`: chiều hidden FAFA; `L`: số query tokens; `D`: chiều projected token.
- `b`: byte/phần tử lưu trữ, bằng 2 cho FP16.

Dung lượng person index cũ: `G * K_max * H * b`.
Dung lượng person index mới: `N * H * b + 8 * (G + 1)` cho offsets.
Mask bool nhỏ vẫn được giữ để kiểm tra sự khớp với DINO.

Chỉ cần một ảnh có nhiều detection, `K_max` có thể lớn hơn nhiều số người
trung bình. Số byte tensor person bỏ được xấp xỉ:

```text
(G * K_max - N) * H * b
```

Projected target tokens vẫn cần `N * L * D * b`. Tất cả person và token được
giữ lại. Precision và công thức model không thay đổi.

Peak dữ liệu tensor cũ xấp xỉ:

```text
N * L * D * b + N * H * b + G * K_max * H * b + G * scene_dim * 4
```

Peak mới xấp xỉ:

```text
N * L * D * b + 2 * N * H * b + G * K_max + 8 * (G + 1)
```

Cả hai còn có metadata/file overhead và biên dự phòng. Peak mới vẫn bao gồm
cả pooled shard lẫn compact index khi ghép. Không thể suy ra chính xác số GiB
sau sửa từ log đã gửi vì log chưa có `N` và `K_max` của detection cache thực.
Log mới in số ảnh đã xong, tổng person, index cũ → packed, target tokens và
peak estimate để kiểm tra ngay trên Kaggle.

## Đo cấu trúc lưu trữ trên dữ liệu giả lập

Đo bằng hai tiến trình Python 3.12 / PyTorch 2.5.1 CPU độc lập, lưu qua
`torch.save`. Ví dụ có 2.048 ảnh, 96 slot/ảnh, 4 person thực/ảnh, hidden 768,
FP16. Đây là ví dụ kiểm tra cấu trúc lưu trữ, **không phải số đo cache Kaggle**.

| Chỉ số | Dense cũ | Packed mới |
| --- | ---: | ---: |
| Tensor person | 288,00 MiB | 12,00 MiB |
| File index gồm mask và offsets cần thiết | 288,19 MiB | 12,20 MiB |
| Peak RSS cả tiến trình, gồm PyTorch | 520,90 MiB | 244,40 MiB |

Tensor person trong ví dụ giảm **24 lần**. Tỷ lệ thực tế phụ thuộc phân bố
detection của gallery; target tokens không giảm theo tỷ lệ này.

## Kiểm tra đã thực hiện

- Ruff toàn `src tools tests labelstudio`, `git diff --check`, `bash -n`.
- Toàn bộ bộ kiểm tra CPU: **544 passed, 18 skipped**. 8 skip do thiếu
  OpenAI CLIP trong môi trường audit; 10 skip cần CUDA.
- Kiểm tra packed và dense format 4/5 cho cùng person, cùng crop order,
  cùng output coarse; có ảnh rỗng, mask không liên tiếp, ảnh lặp trong batch.
- Kiểm tra padding lớn không biến thành tensor person lớn trong index mới.
- Kiểm tra dừng giữa extraction và dừng lúc publish index: tiếp tục được,
  giữ cache ID và shard hợp lệ, không re-extract ảnh đã hoàn thành.
- Kiểm tra offsets sai bị từ chối; file dư/hỏng không giảm mức dự phòng đĩa.
- Kiểm tra target shard được đọc một lần cho mỗi ảnh và được giải phóng trước
  shard kế tiếp; các vị trí ảnh lặp nhận cùng dữ liệu.
- Các integration test hiện có tiếp tục chạy build → train → retrieve → evaluate,
  query text edit/reorder và nguồn read-only bằng encoder giả lập.

Chưa chạy official FAFA trên T4 hoặc build lại toàn bộ 37.107 ảnh. Các kiểm tra
CPU dùng backbone nhỏ/FAFA giả lập, không xác nhận tốc độ, VRAM hay điểm retrieval
thực tế trên Kaggle.

## Áp dụng và chạy tiếp

Giải nén gói patch tại `/kaggle/working`, sau đó:

```bash
cd /kaggle/working/referential-composition-retrieval
git apply --check /kaggle/working/rcr-fafa-memory-fix/rcr-fafa-memory-fix.patch
git apply /kaggle/working/rcr-fafa-memory-fix/rcr-fafa-memory-fix.patch
```

Patch dành cho base commit ghi ở đầu báo cáo. Nếu `--check` báo conflict,
đối chiếu thay đổi local với các file trong thư mục `files/` của gói trước khi
áp dụng. Config và đường dẫn dataset/checkpoint hiện có không bị sửa bởi patch.

Nếu môi trường và runtime assets đã được prepare như log, chạy trực tiếp:

```bash
.venv-fafa/bin/python tools/run.py build-cache \
  --config configs/proposed.yaml --cache-stage fafa-image

.venv-fafa/bin/python tools/run.py build-cache \
  --config configs/proposed.yaml --cache-stage fafa-query
```

FAFA worker tách riêng và dùng lại DINO cache hiện có. Không cần cài lại cả hai
venv cho lần retry này. Image build format 5 đang dở dùng cùng signature/shard
nên có thể tiếp tục và publish format 6. Cache image format 4/5 đã hoàn tất được
đọc nguyên trạng, không tự chuyển đổi; code cũ không đọc được format 6.

`ModuleNotFoundError: wrapt` từ startup hook và các FutureWarning trong log
không phải exception làm lần chạy này dừng: luồng đã đi qua chúng đến disk guard.
Nếu startup hook vẫn báo thiếu wrapt, kiểm tra Python thực thi hook, gồm cả
`python3` host; việc package có trong `.venv-fafa` không chứng minh host có nó.

Hai lần load native model trong `prepare` và extraction là hai giai đoạn riêng.
EVA/Q-Former runtime assets vẫn phục vụ official loader trước khi nạp tuned
checkpoint; không tự xóa chúng hoặc symlink checkpoint đang dùng. DINO/source
caches vẫn được đọc theo định dạng hiện có; audit này tập trung vào FAFA build
đang lỗi và đường đọc cache của train/retrieve.
