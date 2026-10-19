# YOLO Hangul Text Detection

YOLOv8n으로 한글 손글씨의 **위치를 바운딩 박스로 탐지**하는 프로젝트입니다. 한글 글자를 읽는 OCR이나 문자 내용 분류는 포함하지 않습니다.

원본 [Hangeul.ipynb](https://github.com/pod369/Image-processing-4/blob/b2f9a5b94bbe353cce22ce469e358bb2715759ce/Hangeul.ipynb)는 `reference/`에 그대로 보존했습니다. 정리본 `hangul_detection.ipynb`는 설치·압축 해제 로그를 제거하고, 경로 설정·학습·예측·선택적 시험 평가를 구분했습니다.

## 포함한 파일

| 파일 | 역할 |
|---|---|
| `train.py` | 사용자 데이터 YAML로 단일 클래스 YOLOv8n 학습 |
| `predict.py` | 직접 학습한 가중치로 이미지/폴더 탐지, 결과 이미지와 라벨 저장 |
| `evaluate.py` | 별도로 보관한 시험 분할 평가 |
| `hangul_detection.ipynb` | Colab 및 로컬 Jupyter에서 사용할 정리된 실행 노트북 |
| `data.example.yaml` | 데이터 경로와 클래스 이름 설정 예시 |
| `original_training_summary.txt` | 원본 노트북에 기록된 마지막 검증 요약 |

## 먼저 준비할 자료

원본의 `Hangeul.zip`, 이미지/라벨 데이터, `best.pt`는 원본 저장소에 포함되어 있지 않아 가져오지 못했습니다. 직접 준비한 데이터와 학습 결과로 실행해야 합니다. **현재 저장소에 데이터나 새로운 탐지 성능이 있는 것으로 간주하면 안 됩니다.**

다음과 같은 데이터 폴더를 준비합니다.

| 분할 | 이미지 | 라벨 |
|---|---|---|
| 학습 | `hangul_dataset/train/images/` | `hangul_dataset/train/labels/` |
| 검증 | `hangul_dataset/valid/images/` | `hangul_dataset/valid/labels/` |
| 시험(선택) | `hangul_dataset/test/images/` | `hangul_dataset/test/labels/` |

이미지 `sample.jpg`의 라벨은 같은 이름의 `sample.txt`입니다. 객체 하나당 한 줄이며 아래는 **형식 설명용 예시**입니다.

```text
0 0.500000 0.500000 0.200000 0.100000
```

순서: `class_id x_center y_center width height`. 클래스는 `0` 하나입니다. 좌표는 이미지 너비/높이로 나눈 0~1 범위의 값입니다. 같은 이미지에 여러 글씨 영역이 있으면 여러 줄을 작성합니다. 글씨가 없는 이미지에는 빈 라벨 파일을 둡니다.

글씨 영역의 단위를 단어 또는 문장 등으로 먼저 정하고 일관되게 라벨링합니다. 같은 원본 문서의 확대·회전·부분 자르기 이미지는 서로 다른 분할에 섞지 않습니다. 모델 선택에 검증 분할을 사용하고, 시험 분할은 마지막 평가에 남겨둡니다.

`data.example.yaml`을 데이터 폴더에 `data.yaml`로 복사한 뒤 `path`를 실제 데이터셋 **절대 경로**로 바꾸세요. Windows에서도 `D:/datasets/hangul_dataset`처럼 슬래시를 쓰면 편합니다. 시험 데이터를 따로 만들었다면 `test: test/images`의 주석을 해제합니다.

## 로컬 실행

Python 3.11을 권장합니다. 명령은 저장소 루트에서 시작합니다.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py --data /absolute/path/to/hangul_dataset/data.yaml --epochs 100 --imgsz 640
python predict.py --model outputs/train/weights/best.pt --source /absolute/path/to/sample.jpg
python evaluate.py --model outputs/train/weights/best.pt --data /absolute/path/to/hangul_dataset/data.yaml
```

`evaluate.py`는 YAML에 별도 `test` 분할이 있어야 실행됩니다. GPU가 없다면 `--device cpu`를 사용할 수 있습니다. `--model yolov8n.pt`는 처음 실행할 때 사전학습 가중치를 내려받습니다. 이미 보유한 `.pt`의 로컬 경로도 지정할 수 있습니다.

원본 기본 설정은 **YOLOv8n, 100 epochs, 640px**입니다. 정리본의 seed는 42, batch는 16입니다. 빠른 실행 확인은 `--epochs 1 --batch 2`로 직접 수행할 수 있지만 학습 결과를 최종 성능으로 사용하지 마세요.

## Colab 실행

Colab의 코드 셀에서 아래 명령을 실행하거나 정리된 노트북을 엽니다. 데이터셋은 직접 업로드하거나 Google Drive를 연결해 경로를 지정해야 합니다.

```python
!git clone https://github.com/pod369/YOLO-Hangul-Text-Detection.git
%cd /content/YOLO-Hangul-Text-Detection
%pip install -r requirements.txt
!python train.py --data /content/hangul_dataset/data.yaml --epochs 100 --imgsz 640
```

ZIP을 준비할 때 `.venv`, 설치 패키지, 캐시를 포함하지 말고 데이터와 YAML만 넣으세요. 정리된 노트북에는 자동 압축 해제 셀을 넣지 않았습니다.

## 원본에 남아 있는 실제 결과

아래는 **원본 노트북의 `best.pt` 검증 로그**입니다. 이번 정리 작업에서 새로 실행하거나 재현한 수치가 아닙니다. 원래 학습일은 로그에 기록되지 않았고, 정리/확인일은 2026-10-14입니다.

| 평가 데이터 | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---|---|---|
| 원본 검증 2장 / 객체 724개 | 0.862 | 0.714 | 0.786 | 0.416 |

검증 이미지가 2장뿐이므로 일반화 성능을 판단하기에 부족합니다. 이 수치를 별도 시험 성능으로 표기하면 안 됩니다. 데이터셋과 학습 가중치가 없어 현재 구성만으로 원본 결과를 재현할 수 없습니다.

## 실행 후 생성되는 제출 자료

학습 후 `outputs/train/`에는 `weights/best.pt`, `weights/last.pt`, `results.csv`, `results.png` 등 Ultralytics 결과가 생성됩니다. 같은 이름으로 다시 실행하면 `train2`처럼 번호가 붙을 수 있으므로 **콘솔에 출력되는 실제 실행 폴더**를 확인하세요. 예측 결과는 `outputs/predict/`, 별도 시험 평가는 `outputs/test/`에 저장됩니다.

직접 만든 라벨 예시, 탐지 이미지, 학습 곡선, 별도 시험 지표와 데이터 분할 설명을 제출 자료에 추가하면 됩니다. **이번 저장소 정리에서는 모델 학습이나 데이터 다운로드를 수행하지 않았습니다.**

## AI-Ex 포트폴리오

2026년 2학기 AI-Ex 포트폴리오의 개별 프로젝트입니다. 원본 파일은 `reference/`에 보존했으며, 정리한 실행 코드와 원본을 구분했습니다. 원본 기준 커밋은 `b2f9a5b94bbe353cce22ce469e358bb2715759ce`입니다. 원본 재배포 라이선스는 별도로 추가하지 않았습니다.

실행 예시와 결과 설명은 이 저장소의 루트 구조에 맞췄습니다. `results/`에 포함된 측정 기록은 2026-10-14 실행 당시의 자료로, 과거 통합 폴더 경로가 메타데이터에 남아 있을 수 있습니다.

| 관련 프로젝트 |
|---|
| [SIFT Matching and Homography](https://github.com/pod369/SIFT-Matching-and-Homography) |
| [Superpixel and Normalized Cut](https://github.com/pod369/Superpixel-and-Normalized-Cut) |
| [Edge and Contour Detection](https://github.com/pod369/Edge-and-Contour-Detection) |
| [Image and Video Effects](https://github.com/pod369/Image-and-Video-Effects) |
| [MNIST Handwritten Digit Recognition](https://github.com/pod369/MNIST-Handwritten-Digit-Recognition) |
