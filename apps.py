import base64
import os
import sys
import threading
import webbrowser

import cv2
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify, send_from_directory
from insightface.app import FaceAnalysis

# =========================
# 경로 설정 (PyInstaller exe / 소스 실행 모두 대응)
# =========================
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS                      # 내부 리소스 위치
    RUN_DIR = os.path.dirname(sys.executable)    # exe 실제 위치
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RUN_DIR = BASE_DIR

FACE_DIR = os.path.join(RUN_DIR, "faces")        # exe 옆: 사람별 폴더
DATA_DIR = os.path.join(RUN_DIR, "data")         # exe 옆: 얼굴 DB 캐시
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DB_PATH = os.path.join(DATA_DIR, "db.npz")

# 모델: 프로그램 안의 models 폴더 우선, 없으면 사용자 폴더(~/.insightface)
MODEL_DIR = os.path.join(BASE_DIR, "models")
if not os.path.isdir(os.path.join(MODEL_DIR, "models", "buffalo_l")):
    MODEL_DIR = os.path.join(os.path.expanduser("~"), ".insightface")

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
INFO_FILE = "info.txt"    # 사람 폴더 안의 메모(보호자 연락처, 주소, 특이사항 등)

# 유사도 기준 (buffalo_l 코사인 유사도)
#  - 같은 사람: 보통 0.5 이상 (조건이 좋으면 0.7 이상)
#  - 다른 사람: 대부분 0.3 미만
HIGH = 0.50
POSSIBLE = 0.35
TOP_N = 5
MAX_SIDE = 1600           # 큰 휴대폰 사진은 축소 후 검출 (속도)

os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("FACE_DIR:", FACE_DIR)
print("MODEL_DIR:", MODEL_DIR)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


# =========================
# InsightFace 초기화 (검출 + 인식만 사용 → 더 빠름)
# =========================
face_app = FaceAnalysis(
    name='buffalo_l',
    root=MODEL_DIR,
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition'],
)
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace 모델 로딩 완료")


# =========================
# 이미지 읽기 (한글 경로 + 휴대폰 회전정보 대응)
# =========================
def decode_image(data):
    try:
        pil = Image.open(data)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        print("이미지 로딩 실패:", e)
        return None

    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def detect_faces(img):
    """얼굴을 큰 순서대로 반환"""
    try:
        faces = face_app.get(img)
    except Exception as e:
        print("face_app.get 오류:", e)
        return []

    def area(f):
        return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])

    return sorted(faces, key=area, reverse=True)


# =========================
# 얼굴 DB (사진 1장 = 벡터 1개, 바뀐 사진만 다시 계산)
# =========================
db_lock = threading.Lock()
db = {"names": [], "files": [], "vectors": np.zeros((0, 512), np.float32)}
db_failed = []    # 얼굴을 찾지 못한 등록 사진


def file_sig(path):
    st = os.stat(path)
    return f"{st.st_mtime_ns}_{st.st_size}"


def build_db():
    global db, db_failed

    cache = {}
    if os.path.exists(DB_PATH):
        try:
            old = np.load(DB_PATH, allow_pickle=False)
            for f, s, v in zip(old["files"], old["sigs"], old["vectors"]):
                cache[str(f)] = (str(s), v)
        except Exception as e:
            print("기존 DB 읽기 실패 → 새로 생성:", e)

    names, files, sigs, vectors, failed = [], [], [], [], []
    computed = 0

    for person in sorted(os.listdir(FACE_DIR)):
        person_path = os.path.join(FACE_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img_file in sorted(os.listdir(person_path)):
            if not img_file.lower().endswith(IMG_EXT):
                continue

            rel = f"{person}/{img_file}"
            path = os.path.join(person_path, img_file)
            sig = file_sig(path)

            if rel in cache and cache[rel][0] == sig:
                vec = cache[rel][1]
            else:
                img = decode_image(path)
                faces = detect_faces(img) if img is not None else []
                computed += 1
                if not faces:
                    print("얼굴 없음:", rel)
                    failed.append(rel)
                    continue
                vec = faces[0].normed_embedding
                print("등록:", rel)

            names.append(person)
            files.append(rel)
            sigs.append(sig)
            vectors.append(vec)

    vectors = np.array(vectors, np.float32) if vectors else np.zeros((0, 512), np.float32)
    np.savez(DB_PATH, names=np.array(names, dtype=str), files=np.array(files, dtype=str),
             sigs=np.array(sigs, dtype=str), vectors=vectors)

    db = {"names": names, "files": files, "vectors": vectors}
    db_failed = failed
    print(f"DB 준비 완료: {len(set(names))}명 / 사진 {len(files)}장 (새로 계산 {computed}장, 실패 {len(failed)}장)")


def read_info(person):
    path = os.path.join(FACE_DIR, person, INFO_FILE)
    if not os.path.exists(path):
        return ""
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                return f.read().strip()
        except UnicodeDecodeError:
            continue
    return ""


def status():
    return {
        "people": len(set(db["names"])),
        "photos": len(db["files"]),
        "failed": db_failed,
    }


build_db()


# =========================
# 웹 화면
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def get_status():
    return jsonify(status())


@app.route("/reload", methods=["POST"])
def reload_db():
    with db_lock:
        build_db()
    return jsonify(status())


# =========================
# 얼굴 비교
# =========================
@app.route("/compare", methods=["POST"])
def compare():
    file = request.files.get("photo")
    if file is None:
        return jsonify({"error": "사진이 전송되지 않았습니다."})

    # 현장 사진은 디스크에 저장하지 않고 메모리에서만 처리
    img = decode_image(file.stream)
    if img is None:
        return jsonify({"error": "사진 파일을 읽을 수 없습니다. (JPG/PNG 사진인지 확인)"})

    faces = detect_faces(img)
    if not faces:
        return jsonify({"error": "사진에서 얼굴을 찾지 못했습니다. 얼굴이 정면으로 크게 나오게 다시 촬영해 달라고 요청하세요."})

    face = faces[0]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    warnings = []
    if len(faces) > 1:
        warnings.append(f"사진에 얼굴이 {len(faces)}명 있습니다. 가장 크게 나온 얼굴(오른쪽 표시)로 대조했습니다. "
                        "다른 사람이면 대상자만 나오게 다시 촬영하세요.")
    if min(x2 - x1, y2 - y1) < 80 or face.det_score < 0.6:
        warnings.append("얼굴이 작거나 흐립니다. 정확도가 떨어질 수 있으니 가까이서 정면으로 다시 촬영하면 좋습니다.")

    # 사용한 얼굴 잘라서 보여주기
    h, w = img.shape[:2]
    pad = int((x2 - x1) * 0.3)
    crop = img[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]
    ok, buf = cv2.imencode(".jpg", crop)
    crop_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""

    with db_lock:
        names, files, vectors = db["names"], db["files"], db["vectors"]

    if len(vectors) == 0:
        return jsonify({"error": "등록된 사진이 없습니다. faces 폴더에 사람별 폴더를 만들고 사진을 넣은 뒤 [DB 다시 읽기]를 누르세요.",
                        "face": crop_url})

    scores = vectors @ face.normed_embedding

    # 사람별 최고 점수
    best = {}
    for i, name in enumerate(names):
        if name not in best or scores[i] > scores[best[name]]:
            best[name] = i

    ranked = sorted(best.items(), key=lambda kv: scores[kv[1]], reverse=True)[:TOP_N]

    results = []
    for name, i in ranked:
        s = float(scores[i])
        level = "high" if s >= HIGH else "possible" if s >= POSSIBLE else "low"
        results.append({
            "name": name,
            "score": round(s * 100, 1),
            "level": level,
            "image": "/faces/" + files[i],
            "info": read_info(name),
        })

    print("대조 결과:", [(r["name"], r["score"]) for r in results])
    return jsonify({"results": results, "warnings": warnings, "face": crop_url})


# =========================
# 등록 사진 제공
# =========================
@app.route("/faces/<path:filename>")
def face_image(filename):
    return send_from_directory(FACE_DIR, filename)


# =========================
# 실행
# =========================
if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="127.0.0.1", port=5000, debug=False)
