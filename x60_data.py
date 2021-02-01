
from tqdm import tqdm
classes  = ['__label__sống_trẻ', '__label__thời_sự', '__label__công_nghệ', '__label__sức_khỏe', '__label__giáo_dục', '__label__xe_360', '__label__thời_trang', '__label__du_lịch', '__label__âm_nhạc', '__label__xuất_bản', '__label__nhịp_sống', '__label__kinh_doanh', '__label__pháp_luật', '__label__ẩm_thực', '__label__thế_giới', '__label__thể_thao', '__label__giải_trí', '__label__phim_ảnh']

with open("train.txt", 'r', encoding="utf-8") as f, open("trainx60.txt", 'w', encoding="utf-8") as f2:
    for line in tqdm(f):
        line = line.strip().lower().split(" ",1)
        label = line[0]
        text = line[1]
        for _ in range(60):
            f2.write("{} {}\n".format(label, text))

with open("test.txt", 'r', encoding="utf-8") as f, open("testx60.txt", 'w', encoding="utf-8") as f2:
    for line in tqdm(f):
        line = line.strip().lower().split(" ",1)
        label = line[0]
        text = line[1]
        for _ in range(60):
            f2.write("{} {}\n".format(label, text))
