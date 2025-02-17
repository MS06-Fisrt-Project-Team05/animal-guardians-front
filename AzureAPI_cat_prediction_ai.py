from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
import time
from dotenv import load_dotenv
import os

# 한글 폰트 설정 (Windows)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 정보 가져오기
PREDICTION_KEY = os.getenv("AZURE_PREDICTION_KEY")
PREDICTION_ENDPOINT = os.getenv("AZURE_PREDICTION_ENDPOINT")
PROJECT_ID = os.getenv("AZURE_PROJECT_ID")
MODEL_NAME = os.getenv("AZURE_MODEL_NAME")

# API 인증 및 클라이언트 생성
credentials = ApiKeyCredentials(in_headers={"Prediction-key": PREDICTION_KEY})
predictor = CustomVisionPredictionClient(endpoint=PREDICTION_ENDPOINT, credentials=credentials)

# Azure Custom Vision API 요청 함수
def get_predictions_from_azure(image):
    try:
        start_time = time.time()  # API 요청 시작 시간 측정
        
        # 이미지 크기 최적화 (224x224로 축소하여 API 요청 속도 개선)
        image = image.resize((224, 224))

        # 이미지 변환 (바이너리)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # API 요청
        print("Azure Custom Vision API 요청 중...")
        results = predictor.classify_image(PROJECT_ID, MODEL_NAME, img_byte_arr)

        # API 응답 속도 체크
        elapsed_time = time.time() - start_time
        print(f"API 응답 완료 (소요 시간: {elapsed_time:.2f}초)")

        # 예측 결과 저장 (확률 변환)
        predictions = {pred.tag_name: pred.probability * 100 for pred in results.predictions}

        return predictions

    except Exception as e:
        print(f"API 요청 중 오류 발생: {str(e)}")
        return {"Error": "Azure API 요청 실패!"}

# 예측 결과를 시각화
def plot_results(predictions):
    labels = list(predictions.keys())
    values = list(predictions.values())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(labels, values, color="royalblue")
    ax.set_xlim(0, 100)
    ax.set_xlabel("확률 (%)")
    ax.set_title("고양이 피부질환 분석 결과")

    # 확률 값 표시 (한글 깨짐 방지)
    for i, v in enumerate(values):
        ax.text(v + 2, i, f"{v:.1f}%", va="center", fontsize=12, color="black")

    plt.tight_layout()
    return fig

# Gradio 인터페이스 함수
def gradio_interface(image):
    predictions = get_predictions_from_azure(image)
    
    # API 요청 실패 시 메시지 출력
    if "Error" in predictions:
        return predictions["Error"], None
    
    return "분석 완료!", plot_results(predictions)

# Gradio UI 실행
gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil", label="고양이 피부 이미지"),
    outputs=["text", gr.Plot(label="분석 결과")],
    title="고양이 피부질환 분석 AI",
    description="이미지를 업로드하면 피부질환을 분석하여 확률을 그래프로 보여줍니다.",
).launch()
