import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import streamlit as st
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from ultralytics import YOLO


# ============================================================
# SMART PLANT GUARDIAN
# AI + IoT FIELD-DEPLOYABLE SMART FARMING ASSISTANT
# ============================================================

st.set_page_config(
    page_title="Smart Plant Guardian",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# INDIA TIME
# ============================================================

INDIA_TZ = ZoneInfo("Asia/Kolkata")


def india_now():
    return datetime.now(INDIA_TZ)


def current_time():
    return india_now().strftime("%I:%M:%S %p")


def current_datetime():
    return india_now().strftime(
        "%A, %d %B %Y • %I:%M:%S %p"
    )


# ============================================================
# DARK AGRICULTURAL UI
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            linear-gradient(
                rgba(4, 32, 20, 0.95),
                rgba(5, 48, 28, 0.97)
            ),
            url("https://images.unsplash.com/photo-1497250681960-ef046c08a56e?auto=format&fit=crop&w=2400&q=80");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        color: #ffffff;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    body,
    p,
    span,
    label,
    div {
        color: #f5fff7;
    }

    h1,
    h2,
    h3,
    h4,
    h5 {
        color: #ffffff !important;
    }

    .stMarkdown {
        color: #f5fff7;
    }

    [data-testid="stAppViewContainer"] h1 {
        color: #ffffff !important;
        font-size: 2.4rem !important;
        font-weight: 850 !important;
        margin-bottom: 0.2rem !important;
    }

    [data-testid="stAppViewContainer"] h2 {
        color: #e1ffe9 !important;
    }

    .section-heading {
        font-size: 1.45rem;
        font-weight: 850;
        color: #ffffff !important;

        margin-top: 30px;
        margin-bottom: 14px;

        padding: 10px 14px;

        border-left: 5px solid #7ee69c;

        background:
            rgba(6, 58, 32, 0.75);

        border-radius: 8px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            rgba(7, 52, 30, 0.90) !important;

        border:
            1px solid rgba(183, 255, 199, 0.23) !important;

        border-radius: 16px !important;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.23);
    }

    [data-testid="stMetricLabel"] {
        color: #c7f5d2 !important;
        font-weight: 750 !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 850 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #a9f2bb !important;
    }

    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #bce9c8 !important;
    }

    input,
    textarea {
        background-color: #f4fff6 !important;
        color: #102719 !important;

        border: 2px solid #76c98d !important;
        border-radius: 10px !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: #607566 !important;
    }

    [data-baseweb="select"] > div {
        background-color: #f4fff6 !important;
        color: #102719 !important;

        border: 2px solid #76c98d !important;
        border-radius: 10px !important;
    }

    [data-baseweb="select"] span {
        color: #102719 !important;
    }

    .stButton > button {
        background:
            linear-gradient(
                135deg,
                #2e8b57,
                #49b96f
            ) !important;

        color: #ffffff !important;

        border: 1px solid #83e6a0 !important;
        border-radius: 12px !important;

        font-weight: 800 !important;

        padding: 0.55rem 1.2rem !important;

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.28);
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #3aa968,
                #65d886
            ) !important;

        color: #ffffff !important;
    }

    [data-testid="stFileUploader"] {
        background:
            rgba(8, 55, 32, 0.94) !important;

        border:
            2px dashed #7de59b !important;

        border-radius: 16px !important;

        padding: 12px !important;
    }

    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #0c4a2b !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }

    [data-testid="stAlert"] {
        border-radius: 12px !important;
    }

    [data-testid="stAlert"] p {
        color: #ffffff !important;
    }

    [data-testid="stProgressBar"] > div {
        background-color: #174c2d !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #71df91 !important;
    }

    img {
        border-radius: 14px;
    }

    .recommendation {
        background:
            linear-gradient(
                135deg,
                rgba(17, 90, 49, 0.97),
                rgba(11, 67, 36, 0.97)
            );

        border-left:
            5px solid #7be59a;

        padding: 15px 18px;

        border-radius: 12px;

        margin-top: 10px;

        color: #ffffff !important;

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.22);
    }

    .recommendation * {
        color: #ffffff !important;
    }

    .warning-note {
        background: #594817;

        border-left:
            5px solid #f3cf5b;

        padding: 15px 18px;

        border-radius: 12px;

        color: #fffdf0 !important;
    }

    .warning-note * {
        color: #fffdf0 !important;
    }

    .footer {
        text-align: center;

        color: #b8e7c4 !important;

        padding-top: 30px;

        font-size: 0.9rem;
    }

    *,
    *::before,
    *::after {
        backdrop-filter: none !important;
        filter: none !important;
        transition: none !important;
        animation: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.title("🌱 Smart Plant Guardian")

st.subheader(
    "🌿 AI + IoT Field-Deployable Smart Farming Assistant"
)

st.caption(
    "🌾 Real-Time Crop • Water • Environment • Risk Intelligence"
)


# ============================================================
# BLYNK CONFIGURATION
# ============================================================

BLYNK_URL = "https://blynk.cloud/external/api/getAll"


PIN_MAP = {
    "temperature": "v0",
    "humidity": "v1",
    "soil": "v2",
    "tank": "v3",
    "pump": "v4",
    "alert": "v5",
    "light": "v6",
    "waterlog": "v7",
    "flow": "v8",
    "priority": "v9",
    "rain": "v10",
    "safety_pump": "v11",
}


FLOW_SCALE = 10.0


def get_blynk_token():

    token = ""

    try:
        token = st.secrets.get(
            "BLYNK_AUTH_TOKEN",
            ""
        )
    except Exception:
        pass

    if not token:

        try:
            token = st.secrets.get(
                "BLYNK_TOKEN",
                ""
            )
        except Exception:
            pass

    if not token:

        token = os.getenv(
            "BLYNK_AUTH_TOKEN",
            ""
        )

    if not token:

        token = os.getenv(
            "BLYNK_TOKEN",
            ""
        )

    return str(token).strip()


def get_pin_map():

    mapping = dict(PIN_MAP)

    try:

        custom = st.secrets.get(
            "PIN_MAP",
            {}
        )

        if custom:

            for key, value in custom.items():

                if key in mapping:

                    mapping[key] = str(
                        value
                    ).lower()

    except Exception:
        pass

    return mapping


def get_blynk_data():

    token = get_blynk_token()

    if not token:
        return {}

    try:

        response = requests.get(
            BLYNK_URL,
            params={
                "token": token
            },
            headers={
                "Cache-Control": "no-cache, no-store",
                "Pragma": "no-cache",
            },
            timeout=3,
        )

        response.raise_for_status()

        data = response.json()

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def get_value(data, pin):

    if not data:
        return None

    raw = data.get(pin)

    if raw is None:
        return None

    try:
        return float(raw)
    except Exception:
        return None


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value):

    return max(
        0.0,
        min(
            100.0,
            float(value)
        )
    )


def severity(score):

    if score >= 75:
        return "High"

    if score >= 45:
        return "Moderate"

    if score >= 20:
        return "Mild"

    return "Low"


def risk_direction(score):

    if score >= 75:
        return "Immediate attention"

    if score >= 45:
        return "Monitor closely"

    if score >= 20:
        return "Monitor"

    return "Normal"


# ============================================================
# SENSOR INTERPRETATION
# ============================================================

def temperature_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v >= 40:
        return "Very High", "Heat risk"

    if v >= 35:
        return "High", "Monitor heat"

    if v < 15:
        return "Low", "Cold condition"

    return "Normal", "Normal"


def humidity_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v >= 90:
        return "Very High", "Disease risk"

    if v >= 80:
        return "High", "Monitor humidity"

    if v < 35:
        return "Low", "Dry air"

    return "Normal", "Normal"


def soil_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v >= 3000:
        return "Dry", "Water may be needed"

    if v >= 2400:
        return "Moderate", "Monitor moisture"

    return "Moist", "Adequate moisture"


def tank_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v >= 2500:
        return "High", "Water available"

    if v >= 1000:
        return "Available", "Water available"

    return "Low", "Water shortage"


def light_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v < 500:
        return "Low", "Low light"

    return "Normal", "Adequate light"


def rain_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v < 2500:
        return "Detected", "Avoid irrigation"

    return "Clear", "No rain signal"


def waterlog_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v >= 1000:
        return "Detected", "Stop irrigation"

    return "Normal", "No waterlogging"


def pump_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v > 0:
        return "ON", "Pump active"

    return "OFF", "Pump inactive"


def safety_pump_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    if v > 0:
        return "ON", "Safety pump active"

    return "OFF", "Safety pump inactive"


def flow_status(v):

    if v is None:
        return "Unavailable", "Waiting"

    lpm = v / FLOW_SCALE

    if lpm <= 0.05:
        return "0.00 L/min", "No flow"

    return (
        f"{lpm:.2f} L/min",
        "Flow detected"
    )


# ============================================================
# DROUGHT / WATER STRESS
# ============================================================

def drought_water_stress(
    soil,
    temperature,
    humidity,
    tank,
    rain,
    waterlog
):

    score = 0

    if soil is not None:

        if soil >= 3000:
            score += 55

        elif soil >= 2400:
            score += 40

        elif soil >= 1800:
            score += 20

    if temperature is not None:

        if temperature >= 40:
            score += 20

        elif temperature >= 35:
            score += 15

        elif temperature >= 32:
            score += 8

    if humidity is not None:

        if humidity < 35:
            score += 15

        elif humidity < 50:
            score += 8

    if tank is not None and tank < 1000:
        score += 15

    if rain is not None and rain < 2500:
        score -= 20

    if waterlog is not None and waterlog >= 1000:
        score -= 20

    return int(clamp(score))


# ============================================================
# HEAT STRESS
# ============================================================

def heat_stress(
    temperature,
    humidity
):

    if temperature is None:
        return 0

    score = 0

    if temperature >= 40:
        score += 75

    elif temperature >= 38:
        score += 60

    elif temperature >= 35:
        score += 42

    elif temperature >= 32:
        score += 22

    if humidity is not None:

        if humidity >= 85:
            score += 20

        elif humidity >= 70:
            score += 10

    return int(clamp(score))


# ============================================================
# WATERLOGGING STRESS
# ============================================================

def waterlogging_stress(v):

    if v is None:
        return 0

    if v >= 1800:
        return 100

    if v >= 1400:
        return 80

    if v >= 1000:
        return 60

    if v > 500:
        return 25

    return 0


# ============================================================
# AGRICULTURAL RISK
# ============================================================

def agricultural_risk(
    drought,
    heat,
    waterlog,
    tank,
    disease=0,
    pest=0,
    nutrient=0
):

    tank_risk = 0

    if (
        tank is not None
        and tank < 1000
    ):
        tank_risk = 100

    result = (
        drought * 0.25
        + heat * 0.18
        + waterlog * 0.18
        + tank_risk * 0.10
        + disease * 0.14
        + pest * 0.05
        + nutrient * 0.10
    )

    return int(clamp(result))


# ============================================================
# INTELLIGENT IRRIGATION ENGINE
# ============================================================

def irrigation_recommendation(
    soil,
    tank,
    rain,
    waterlog,
    temperature,
    humidity,
    pump
):

    if (
        waterlog is not None
        and waterlog >= 1000
    ):
        return (
            "DO NOT IRRIGATE",
            "Waterlogging detected. Allow excess field water to drain."
        )

    if (
        rain is not None
        and rain < 2500
    ):
        return (
            "DO NOT IRRIGATE",
            "Rain is detected. Avoid unnecessary irrigation."
        )

    if (
        tank is not None
        and tank < 1000
    ):
        return (
            "WATER UNAVAILABLE",
            "Tank water is too low for normal irrigation."
        )

    if soil is None:
        return (
            "MONITOR",
            "Soil sensor data is unavailable."
        )

    if soil >= 3000:

        if (
            temperature is not None
            and temperature >= 35
        ):
            return (
                "PRIORITY IRRIGATION",
                "Soil is dry and temperature is high."
            )

        return (
            "IRRIGATION RECOMMENDED",
            "Soil is dry and water is available."
        )

    if soil >= 2400:
        return (
            "MONITOR",
            "Soil moisture is moderate."
        )

    return (
        "NO IRRIGATION",
        "Soil currently has adequate moisture."
    )


# ============================================================
# PUMP FAULT DETECTION
# ============================================================

def pump_fault_status(
    pump,
    flow
):

    if (
        pump is None
        or flow is None
    ):
        return (
            "UNKNOWN",
            "Waiting for pump and flow data."
        )

    lpm = flow / FLOW_SCALE

    if (
        pump > 0
        and lpm <= 0.05
    ):
        return (
            "POSSIBLE FAULT",
            "Pump is ON but meaningful water flow is not detected."
        )

    if (
        pump <= 0
        and lpm > 0.05
    ):
        return (
            "CHECK SYSTEM",
            "Flow is detected while the irrigation pump is OFF."
        )

    return (
        "NORMAL",
        "Pump and flow feedback are consistent."
    )


# ============================================================
# DISEASE MODEL
# ============================================================

CLASS_NAMES = [

    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",

    "Blueberry___healthy",

    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",

    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",

    "Orange___Haunglongbing_(Citrus_greening)",

    "Peach___Bacterial_spot",
    "Peach___healthy",

    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",

    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",

    "Raspberry___healthy",

    "Soybean___healthy",

    "Squash___Powdery_mildew",

    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",

    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# IMPORTANT:
# MobileNetV2 confidence below this value means that the
# prediction is not considered a confident match.
#
# < 25%  -> Green Gram visual screening
# >= 25% -> Single best MobileNetV2 prediction
# ============================================================

MOBILENET_MATCH_THRESHOLD = 25.0


@st.cache_resource
def load_disease_model():

    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Sequential(
        nn.Dropout(
            p=0.2
        ),

        nn.Linear(
            model.classifier[1].in_features,
            38
        )
    )

    model.load_state_dict(
        torch.load(
            "model/mobilenetv2_plant.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


transform = transforms.Compose(
    [

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        ),

    ]
)


def clean_label(name):

    return (
        name
        .replace(
            "___",
            " — "
        )
        .replace(
            "_",
            " "
        )
    )


def disease_confidence(conf):

    if conf >= 80:
        return "High confidence"

    if conf >= 60:
        return "Moderate confidence"

    if conf >= 40:
        return "Low confidence — verify visually"

    return (
        "Very low confidence — prediction uncertain"
    )


def disease_direction(
    label,
    confidence
):

    label_lower = label.lower()

    if "healthy" in label_lower:

        return (
            "Direction: No obvious disease pattern detected. "
            "Continue regular crop monitoring."
        )

    if confidence < 50:

        return (
            "Direction: Prediction is uncertain. "
            "Capture a clearer close-up leaf image and verify the symptom visually."
        )

    return (
        "Direction: Inspect affected leaves closely, "
        "check nearby plants, isolate visibly affected plants when appropriate, "
        "and confirm the suspected disease before treatment."
    )


def disease_predict(image):

    try:

        model = load_disease_model()

        x = transform(
            image.convert(
                "RGB"
            )
        ).unsqueeze(0)

        with torch.no_grad():

            output = model(x)

            probabilities = torch.softmax(
                output,
                dim=1
            )[0]

        best_index = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                best_index
            ].item() * 100
        )

        return (
            CLASS_NAMES[
                best_index
            ],
            confidence
        )

    except Exception:

        return (
            None,
            0.0
        )


# ============================================================
# MOBILE NET MATCH / GREEN GRAM FALLBACK
# ============================================================

def should_use_green_gram_fallback(
    disease_label,
    disease_conf
):
    """
    MobileNetV2 is checked FIRST.

    If the best MobileNetV2 prediction has confidence
    below 25%, the image is treated as not having a
    sufficiently confident match to the supported
    MobileNetV2 classes.

    Then Green Gram visual screening is used as the
    fallback screening route.

    IMPORTANT:
    This does NOT prove that the image is Green Gram.
    It means the image was not confidently matched by
    the 38-class MobileNetV2 model.
    """

    if disease_label is None:
        return True

    return disease_conf < MOBILENET_MATCH_THRESHOLD


# ============================================================
# GREEN GRAM AUTOMATIC VISUAL SCREENING
# ============================================================

def green_gram_health_screening(image):

    arr = np.asarray(
        image.convert(
            "RGB"
        ).resize(
            (224, 224)
        )
    ).astype(
        np.float32
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green = (
        (g > r * 1.05)
        &
        (g > b * 1.03)
    )

    yellow = (
        (r > b * 1.18)
        &
        (g > b * 1.12)
        &
        (r > 70)
    )

    brown = (
        (r > g * 1.12)
        &
        (g > b * 1.05)
        &
        (r > 70)
        &
        (g < 170)
    )

    green_ratio = float(
        green.mean()
    )

    yellow_ratio = float(
        yellow.mean()
    )

    brown_ratio = float(
        brown.mean()
    )

    stress_score = int(
        clamp(
            yellow_ratio * 110
            +
            brown_ratio * 120
            -
            green_ratio * 25
        )
    )

    if stress_score < 20:

        status = (
            "🟢 Healthy-looking Green Gram"
        )

        recommendation = (
            "Continue regular monitoring, irrigation "
            "and field inspection."
        )

    elif stress_score < 50:

        status = (
            "🟡 Possible Green Gram Stress"
        )

        recommendation = (
            "Inspect the leaves for yellowing, spots, "
            "curling or pest activity and monitor soil "
            "moisture and environmental conditions."
        )

    else:

        status = (
            "🔴 High Visible Stress"
        )

        recommendation = (
            "Inspect affected leaves and nearby plants "
            "carefully. Check for disease, pests, nutrient "
            "problems and water stress before taking "
            "treatment action."
        )

    return (
        status,
        stress_score,
        recommendation
    )


# ============================================================
# PEST MODEL
# ============================================================

@st.cache_resource
def load_pest_model():

    return YOLO(
        "model/best.pt"
    )


def pest_predict(image):

    try:

        model = load_pest_model()

        results = model.predict(
            source=np.array(
                image.convert(
                    "RGB"
                )
            ),
            conf=0.25,
            verbose=False
        )

        detections = []

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0] * 100
                )

                name = result.names[
                    class_id
                ]

                detections.append(
                    (
                        name,
                        confidence
                    )
                )

        detections.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return detections

    except Exception:

        return []


# ============================================================
# VISUAL NUTRIENT-STRESS ASSESSMENT
# ============================================================

def nutrient_assessment(image):

    arr = np.asarray(
        image.convert(
            "RGB"
        ).resize(
            (224, 224)
        )
    ).astype(
        np.float32
    )

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    green = (
        (g > r * 1.05)
        &
        (g > b * 1.03)
    )

    yellow = (
        (r > b * 1.18)
        &
        (g > b * 1.12)
        &
        (r > 70)
    )

    brown = (
        (r > g * 1.12)
        &
        (g > b * 1.05)
        &
        (r > 70)
        &
        (g < 170)
    )

    green_ratio = float(
        green.mean()
    )

    yellow_ratio = float(
        yellow.mean()
    )

    brown_ratio = float(
        brown.mean()
    )

    score = int(
        clamp(
            yellow_ratio * 120
            +
            brown_ratio * 100
            -
            green_ratio * 20
        )
    )

    if score >= 65:
        level = "High visible stress"

    elif score >= 40:
        level = "Moderate visible stress"

    elif score >= 20:
        level = "Mild visible stress"

    else:
        level = "Low visible stress"

    if (
        yellow_ratio > 0.22
        and green_ratio < 0.50
    ):

        pattern = (
            "Possible nitrogen-related chlorosis"
        )

    elif (
        yellow_ratio > 0.16
        and green_ratio < 0.58
    ):

        pattern = (
            "Possible magnesium/iron-related chlorosis"
        )

    elif brown_ratio > 0.18:

        pattern = (
            "Possible potassium-related / edge-burn stress"
        )

    elif score >= 20:

        pattern = (
            "General visible nutrient-stress pattern"
        )

    else:

        pattern = (
            "No obvious nutrient-stress pattern"
        )

    return (
        score,
        level,
        pattern
    )


# ============================================================
# FARMER RECOMMENDATION ENGINE
# ============================================================

def generate_farmer_recommendation(
    soil,
    tank,
    rain,
    waterlog,
    temperature,
    humidity,
    drought,
    heat,
    risk,
    disease_label=None,
    disease_conf=0,
    pests=None,
    nutrient_score=0
):

    recommendations = []

    irrigation, irrigation_reason = (
        irrigation_recommendation(
            soil,
            tank,
            rain,
            waterlog,
            temperature,
            humidity,
            0
        )
    )

    if irrigation == "PRIORITY IRRIGATION":

        recommendations.append(
            "💧 Prioritize irrigation because the soil is dry and heat stress is elevated."
        )

    elif irrigation == "IRRIGATION RECOMMENDED":

        recommendations.append(
            "💧 Irrigation can be considered because the soil is dry and water is available."
        )

    elif irrigation == "DO NOT IRRIGATE":

        recommendations.append(
            "🌧️ Do not irrigate now because rain or excess-water conditions are detected."
        )

    elif irrigation == "WATER UNAVAILABLE":

        recommendations.append(
            "🚰 Check or refill the irrigation water source before operating the pump."
        )

    else:

        recommendations.append(
            "💧 Continue monitoring soil moisture before irrigating."
        )

    if heat >= 60:

        recommendations.append(
            "🌡️ Heat stress is high. Protect the crop from prolonged heat and monitor moisture closely."
        )

    elif heat >= 35:

        recommendations.append(
            "🌡️ Moderate heat stress is present. Monitor crop temperature and water availability."
        )

    if drought >= 60:

        recommendations.append(
            "🏜️ Water stress is high. Check soil moisture and available irrigation water."
        )

    elif drought >= 35:

        recommendations.append(
            "🏜️ Some water stress is developing. Increase monitoring frequency."
        )

    if waterlog_status(
        waterlog
    )[0] == "Detected":

        recommendations.append(
            "🌊 Waterlogging is detected. Stop irrigation and improve drainage."
        )

    if disease_label:

        if "healthy" not in disease_label.lower():

            if disease_conf >= 60:

                recommendations.append(
                    "🦠 A disease pattern was detected. Inspect affected leaves and confirm before treatment."
                )

            else:

                recommendations.append(
                    "🦠 Disease prediction is uncertain. Capture a clearer leaf image and verify visually."
                )

    if pests:

        recommendations.append(
            "🐛 Possible pest activity was detected. Inspect the affected plant area before control action."
        )

    if nutrient_score >= 40:

        recommendations.append(
            "🌿 Visible nutrient-stress symptoms are present. Confirm with appropriate plant or soil testing before fertilizer application."
        )

    if risk >= 75:

        recommendations.append(
            "🚨 Overall agricultural risk is high. Give immediate attention to the main stress signals."
        )

    elif risk >= 45:

        recommendations.append(
            "⚠️ Overall agricultural risk is moderate. Monitor the crop and sensors closely."
        )

    else:

        recommendations.append(
            "✅ Overall agricultural risk is currently relatively low. Continue routine monitoring."
        )

    return recommendations


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "live" not in st.session_state:

    st.session_state.live = {

        "data": {},

        "temperature": None,
        "humidity": None,
        "soil": None,
        "tank": None,

        "pump": None,
        "alert": None,
        "light": None,

        "waterlog": None,
        "flow": None,
        "priority": None,
        "rain": None,

        "safety_pump": None,

        "drought": 0,
        "heat": 0,
        "waterlog_score": 0,

        "agricultural_risk": 0,

        "last_update": "Waiting",
    }


# ============================================================
# LIVE MONITOR
# ============================================================

@st.fragment(run_every=1)
def live_monitor():

    pin = get_pin_map()

    data = get_blynk_data()

    temperature = get_value(
        data,
        pin["temperature"]
    )

    humidity = get_value(
        data,
        pin["humidity"]
    )

    soil = get_value(
        data,
        pin["soil"]
    )

    tank = get_value(
        data,
        pin["tank"]
    )

    pump = get_value(
        data,
        pin["pump"]
    )

    alert = get_value(
        data,
        pin["alert"]
    )

    light = get_value(
        data,
        pin["light"]
    )

    waterlog = get_value(
        data,
        pin["waterlog"]
    )

    flow = get_value(
        data,
        pin["flow"]
    )

    priority = get_value(
        data,
        pin["priority"]
    )

    rain = get_value(
        data,
        pin["rain"]
    )

    safety_pump = get_value(
        data,
        pin["safety_pump"]
    )

    drought = drought_water_stress(
        soil,
        temperature,
        humidity,
        tank,
        rain,
        waterlog
    )

    heat = heat_stress(
        temperature,
        humidity
    )

    waterlog_score = waterlogging_stress(
        waterlog
    )

    agricultural_score = agricultural_risk(
        drought,
        heat,
        waterlog_score,
        tank
    )

    st.session_state.live = {

        "data": data,

        "temperature": temperature,
        "humidity": humidity,
        "soil": soil,
        "tank": tank,

        "pump": pump,
        "alert": alert,
        "light": light,

        "waterlog": waterlog,
        "flow": flow,
        "priority": priority,
        "rain": rain,

        "safety_pump": safety_pump,

        "drought": drought,
        "heat": heat,
        "waterlog_score": waterlog_score,

        "agricultural_risk": agricultural_score,

        "last_update": current_time(),
    }


    # ========================================================
    # SYSTEM TIME
    # ========================================================

    st.markdown(
        '<div class="section-heading">🕒 System Time</div>',
        unsafe_allow_html=True
    )

    clock_col1, clock_col2 = st.columns(
        [2, 1]
    )

    with clock_col1:

        st.metric(
            "⏱️ Current System Time",
            current_datetime()
        )

    with clock_col2:

        st.metric(
            "📡 Last Blynk Update",
            st.session_state.live[
                "last_update"
            ]
        )


    # ========================================================
    # LIVE FARM MONITORING
    # ========================================================

    st.markdown(
        '<div class="section-heading">📡 Live Farm Monitoring</div>',
        unsafe_allow_html=True
    )

    row1 = st.columns(5)

    sensor_items = [

        (
            "🌡️ Temperature",
            (
                f"{temperature:.1f} °C"
                if temperature is not None
                else "N/A"
            ),
            temperature_status(
                temperature
            )
        ),

        (
            "💧 Humidity",
            (
                f"{humidity:.1f} %"
                if humidity is not None
                else "N/A"
            ),
            humidity_status(
                humidity
            )
        ),

        (
            "🌱 Soil Sensor",
            (
                f"{soil:.0f}"
                if soil is not None
                else "N/A"
            ),
            soil_status(
                soil
            )
        ),

        (
            "🚰 Tank Water",
            (
                "N/A"
                if tank is None
                else tank_status(
                    tank
                )[0]
            ),
            tank_status(
                tank
            )
        ),

        (
            "⚙️ Irrigation Pump",
            (
                "N/A"
                if pump is None
                else pump_status(
                    pump
                )[0]
            ),
            pump_status(
                pump
            )
        ),

    ]


    for col, item in zip(
        row1,
        sensor_items
    ):

        title, value, status = item

        with col:

            with st.container(
                border=True
            ):

                st.metric(
                    title,
                    value
                )

                st.caption(
                    status[0]
                )

                st.caption(
                    status[1]
                )


    # ========================================================
    # SECOND SENSOR ROW
    # ========================================================

    row2 = st.columns(5)

    second_items = [

        (
            "☀️ Light Sensor",
            (
                "N/A"
                if light is None
                else f"{light:.0f}"
            ),
            light_status(
                light
            )
        ),

        (
            "🌧️ Rain Sensor",
            (
                "N/A"
                if rain is None
                else rain_status(
                    rain
                )[0]
            ),
            rain_status(
                rain
            )
        ),

        (
            "🌊 Waterlogging",
            (
                "N/A"
                if waterlog is None
                else waterlog_status(
                    waterlog
                )[0]
            ),
            waterlog_status(
                waterlog
            )
        ),

        (
            "🚿 Water Flow",
            flow_status(
                flow
            )[0],
            flow_status(
                flow
            )
        ),

        (
            "🛡️ Safety Pump",
            (
                "N/A"
                if safety_pump is None
                else safety_pump_status(
                    safety_pump
                )[0]
            ),
            safety_pump_status(
                safety_pump
            )
        ),

    ]


    for col, item in zip(
        row2,
        second_items
    ):

        title, value, status = item

        with col:

            with st.container(
                border=True
            ):

                st.metric(
                    title,
                    value
                )

                st.caption(
                    status[0]
                )

                st.caption(
                    status[1]
                )


    # ========================================================
    # THIRD LIVE STATUS ROW
    # ========================================================

    row3 = st.columns(3)

    with row3[0]:

        st.metric(
            "🎯 Irrigation Priority",
            (
                f"{priority:.0f}/100"
                if priority is not None
                else "N/A"
            )
        )


    with row3[1]:

        pump_fault, pump_fault_reason = (
            pump_fault_status(
                pump,
                flow
            )
        )

        st.metric(
            "🔧 Pump Health",
            pump_fault
        )

        st.caption(
            pump_fault_reason
        )


    with row3[2]:

        st.metric(
            "🔔 ESP32 Alert",
            (
                "Active"
                if (
                    alert is not None
                    and alert > 0
                )
                else "Normal"
                if alert is not None
                else "N/A"
            )
        )


    # ========================================================
    # INTELLIGENT IRRIGATION
    # ========================================================

    irrigation, irrigation_reason = (
        irrigation_recommendation(
            soil,
            tank,
            rain,
            waterlog,
            temperature,
            humidity,
            pump
        )
    )

    st.markdown(
        '<div class="section-heading">💧 Intelligent Irrigation Decision</div>',
        unsafe_allow_html=True
    )

    with st.container(
        border=True
    ):

        st.subheader(
            irrigation
        )

        st.write(
            irrigation_reason
        )


    # ========================================================
    # LIVE AGRICULTURAL RISK
    # ========================================================

    st.markdown(
        '<div class="section-heading">📊 Live Agricultural Risk Intelligence</div>',
        unsafe_allow_html=True
    )

    risk_cols = st.columns(4)

    live_risks = [

        (
            risk_cols[0],
            "🏜️ Drought / Water Stress",
            drought
        ),

        (
            risk_cols[1],
            "🌡️ Heat Stress",
            heat
        ),

        (
            risk_cols[2],
            "🌊 Waterlogging Stress",
            waterlog_score
        ),

        (
            risk_cols[3],
            "📊 Agricultural Risk",
            agricultural_score
        ),

    ]


    for col, title, score in live_risks:

        with col:

            with st.container(
                border=True
            ):

                st.metric(
                    title,
                    f"{score}/100"
                )

                st.progress(
                    int(score)
                )

                st.caption(
                    f"{severity(score)} • "
                    f"{risk_direction(score)}"
                )


    if not data:

        st.warning(
            "📡 Blynk data is currently unavailable. "
            "The dashboard is waiting for live ESP32 data."
        )


live_monitor()


# ============================================================
# SMART DECISION FLOW
# ============================================================

st.markdown(
    '<div class="section-heading">🧠 Smart Decision Flow</div>',
    unsafe_allow_html=True
)

flow_cols = st.columns(5)

flow_steps = [

    "📡 MONITOR",
    "🔬 ANALYZE",
    "🧠 DECIDE",
    "⚙️ ACT",
    "🔔 ALERT",
]


for col, step in zip(
    flow_cols,
    flow_steps
):

    with col:
        st.info(step)


# ============================================================
# AI CROP & LEAF ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-heading">🔬 AI Crop & Leaf Analysis</div>',
    unsafe_allow_html=True
)

st.write(
    "🌿 Upload a clear crop or leaf image for "
    "AI-assisted crop-health screening, disease analysis, "
    "pest detection and visual nutrient-stress assessment."
)

st.caption(
    "🧠 AI flow: MobileNetV2 is checked first. "
    "If its best match is below 25% confidence, "
    "the system switches to Green Gram visual screening."
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "📤 Upload Crop / Leaf Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    key="crop_image_upload"
)


disease_label = None
disease_conf = 0.0

pests = []

nutrient_score = 0

nutrient_level = "Not analyzed"

nutrient_pattern = (
    "Upload an image to analyze."
)

green_gram_detected = False

green_gram_status = None

green_gram_score = 0

green_gram_recommendation = None


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded is not None:

    try:

        image = Image.open(
            uploaded
        ).convert(
            "RGB"
        )

        image_col, result_col = (
            st.columns(
                [1, 1.35]
            )
        )


        with image_col:

            st.subheader(
                "🌿 Uploaded Crop / Leaf"
            )

            st.image(
                image,
                caption="Crop image selected for AI analysis",
                use_container_width=True
            )


        # ====================================================
        # STEP 1 — MOBILENETV2 FIRST
        # ====================================================

        disease_label, disease_conf = (
            disease_predict(
                image
            )
        )


        # ====================================================
        # STEP 2 — CHECK 25% THRESHOLD
        # ====================================================

        green_gram_detected = (
            should_use_green_gram_fallback(
                disease_label,
                disease_conf
            )
        )


        with result_col:

            # =================================================
            # MOBILE NET LOW CONFIDENCE
            # → GREEN GRAM FALLBACK
            # =================================================

            if green_gram_detected:

                st.subheader(
                    "🌱 Green Gram AI-Assisted Health Screening"
                )

                if disease_label is None:

                    st.info(
                        "ℹ️ MobileNetV2 could not produce a valid prediction. "
                        "The system is switching to Green Gram visual screening."
                    )

                else:

                    st.info(
                        f"ℹ️ MobileNetV2 best match was "
                        f"{disease_conf:.1f}% confidence, "
                        f"which is below the {MOBILENET_MATCH_THRESHOLD:.0f}% "
                        "match threshold. The system is switching to "
                        "Green Gram visual screening."
                    )


                (
                    green_gram_status,
                    green_gram_score,
                    green_gram_recommendation
                ) = green_gram_health_screening(
                    image
                )


                if green_gram_score < 20:

                    st.success(
                        green_gram_status
                    )

                elif green_gram_score < 50:

                    st.warning(
                        green_gram_status
                    )

                else:

                    st.error(
                        green_gram_status
                    )


                st.metric(
                    "🌿 Green Gram Visible Stress Score",
                    f"{green_gram_score}/100"
                )

                st.progress(
                    green_gram_score
                )

                st.write(
                    green_gram_recommendation
                )

                st.caption(
                    "AI-assisted visual screening based on "
                    "visible leaf characteristics. This fallback "
                    "does not prove that the uploaded image is Green Gram. "
                    "Confirm crop identity and suspected disease, pest "
                    "or nutrient problems through field inspection."
                )

                # Do not use low-confidence MobileNet prediction
                # as the final disease result.
                disease_label = None
                disease_conf = 0.0


            # =================================================
            # MOBILE NET CONFIDENT MATCH
            # =================================================

            else:

                st.subheader(
                    "🦠 Leaf Disease Prediction"
                )

                if disease_label:

                    readable = clean_label(
                        disease_label
                    )

                    st.success(
                        f"🌿 Best MobileNetV2 Match: {readable}"
                    )

                    st.metric(
                        "🎯 Best Prediction Confidence",
                        f"{disease_conf:.1f}%"
                    )

                    st.caption(
                        disease_confidence(
                            disease_conf
                        )
                    )

                    st.write(
                        disease_direction(
                            readable,
                            disease_conf
                        )
                    )

                    st.caption(
                        f"MobileNetV2 accepted this best match "
                        f"because confidence is ≥ "
                        f"{MOBILENET_MATCH_THRESHOLD:.0f}%."
                    )

                else:

                    st.error(
                        "Disease model could not analyze this image. "
                        "Check that model/mobilenetv2_plant.pth exists "
                        "and matches the 38-class model architecture."
                    )


            # =================================================
            # PEST DETECTION
            # =================================================

            st.subheader(
                "🐛 Pest Detection"
            )

            pests = pest_predict(
                image
            )

            if pests:

                for (
                    pest_name,
                    confidence
                ) in pests:

                    st.write(
                        f"**🐛 {pest_name}** — "
                        f"{confidence:.1f}%"
                    )

                    st.progress(
                        min(
                            int(confidence),
                            100
                        )
                    )

                    if confidence >= 60:

                        st.caption(
                            "Detected — verify visually."
                        )

                    else:

                        st.caption(
                            "Low-confidence detection — verify visually."
                        )

            else:

                st.success(
                    "✅ No pest detected at the current detection threshold."
                )


        # ====================================================
        # NUTRIENT STRESS
        # ====================================================

        (
            nutrient_score,
            nutrient_level,
            nutrient_pattern
        ) = nutrient_assessment(
            image
        )

        st.subheader(
            "🌿 Visual Nutrient-Stress Assessment"
        )

        nutrient_cols = st.columns(
            3
        )

        with nutrient_cols[0]:

            with st.container(
                border=True
            ):

                st.metric(
                    "Visual Stress Score",
                    f"{nutrient_score}/100"
                )

                st.progress(
                    nutrient_score
                )

        with nutrient_cols[1]:

            with st.container(
                border=True
            ):

                st.metric(
                    "Stress Level",
                    nutrient_level
                )

        with nutrient_cols[2]:

            with st.container(
                border=True
            ):

                st.write(
                    "**Possible Visual Pattern**"
                )

                st.write(
                    nutrient_pattern
                )


        st.warning(
            "🌿 Visual nutrient assessment is a screening tool. "
            "It does not directly measure soil or leaf nutrient concentration. "
            "Confirm suspected nutrient deficiency with appropriate agricultural testing."
        )


    except Exception as error:

        st.error(
            f"Image analysis could not be completed: {error}"
        )


# ============================================================
# FINAL RISK ENGINE
# ============================================================

live = st.session_state.live

temperature = live["temperature"]

humidity = live["humidity"]

soil = live["soil"]

tank = live["tank"]

rain = live["rain"]

waterlog = live["waterlog"]

drought = live["drought"]

heat = live["heat"]

waterlog_score = live["waterlog_score"]


highest_pest_confidence = max(
    [
        p[1]
        for p in pests
    ],
    default=0
)


combined_risk = agricultural_risk(
    drought,
    heat,
    waterlog_score,
    tank,

    disease_conf
    if disease_label
    else 0,

    highest_pest_confidence,

    nutrient_score
)


# ============================================================
# FINAL AGRICULTURAL RISK SCORE
# ============================================================

st.markdown(
    '<div class="section-heading">🚨 Final Agricultural Risk Score</div>',
    unsafe_allow_html=True
)

risk_cols = st.columns(4)

final_scores = [

    (
        risk_cols[0],
        "🏜️ Drought / Water Stress",
        drought
    ),

    (
        risk_cols[1],
        "🌡️ Heat Stress",
        heat
    ),

    (
        risk_cols[2],
        "🌿 Nutrient Visual Stress",
        nutrient_score
    ),

    (
        risk_cols[3],
        "🚨 Agricultural Risk",
        combined_risk
    ),

]


for col, title, score in final_scores:

    with col:

        with st.container(
            border=True
        ):

            st.metric(
                title,
                f"{score}/100"
            )

            st.progress(
                int(score)
            )

            st.caption(
                f"{severity(score)} • "
                f"{risk_direction(score)}"
            )


# ============================================================
# FARMER RECOMMENDATIONS
# ============================================================

st.markdown(
    '<div class="section-heading">👨‍🌾 Farmer Recommendation Engine</div>',
    unsafe_allow_html=True
)

recommendations = (
    generate_farmer_recommendation(
        soil=soil,
        tank=tank,
        rain=rain,
        waterlog=waterlog,
        temperature=temperature,
        humidity=humidity,
        drought=drought,
        heat=heat,
        risk=combined_risk,
        disease_label=disease_label,
        disease_conf=disease_conf,
        pests=pests,
        nutrient_score=nutrient_score
    )
)


for recommendation in recommendations:

    st.markdown(
        f"""
        <div class="recommendation">
            {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FARMER HELP ASSISTANT
# ============================================================

st.markdown(
    '<div class="section-heading">🌱 Farmer Help Assistant</div>',
    unsafe_allow_html=True
)

st.write(
    "👨‍🌾 Select a common question or type your own question."
)


questions = [

    "Should I irrigate now?",

    "Is my soil too dry?",

    "Is water available?",

    "Is my crop under heat stress?",

    "Is waterlogging detected?",

    "Is rain detected?",

    "Is there a pump fault?",

    "What does the disease prediction mean?",

    "Are pests detected?",

    "Is there visible nutrient stress?",

    "What is the drought / water stress?",

    "What is the Agricultural Risk Score?",

    "What should I do now?",

]


selected_question = st.selectbox(
    "1️⃣ Select a common question",
    [
        "Choose a question..."
    ]
    +
    questions,
    key="farmer_common_question"
)


typed_question = st.text_input(
    "2️⃣ Type your own question",
    placeholder=(
        "Example: Should I water my crop now?"
    ),
    key="farmer_typed_question"
)


ask = st.button(
    "🌱 Ask Farmer Assistant",
    type="primary",
    key="farmer_assistant_button"
)


if ask:

    question = typed_question.strip()

    if not question:

        if (
            selected_question
            !=
            "Choose a question..."
        ):

            question = selected_question

    if not question:

        st.warning(
            "Please select a question or type your own question."
        )

    else:

        q = question.lower()


        if (
            "irrigat" in q
            or "water now" in q
            or "should i water" in q
        ):

            irrigation, reason = (
                irrigation_recommendation(
                    soil,
                    tank,
                    rain,
                    waterlog,
                    temperature,
                    humidity,
                    live["pump"]
                )
            )

            answer = (
                f"**{irrigation}** — "
                f"{reason}"
            )


        elif "soil" in q:

            soil_state = soil_status(
                soil
            )

            answer = (
                f"**Soil status:** "
                f"{soil_state[0]}. "
                f"{soil_state[1]}"
            )


        elif (
            "tank" in q
            or "water available" in q
            or "water shortage" in q
        ):

            tank_state = tank_status(
                tank
            )

            answer = (
                f"**Water status:** "
                f"{tank_state[0]}. "
                f"{tank_state[1]}"
            )


        elif (
            "heat" in q
            or "temperature" in q
        ):

            answer = (
                f"**Heat-stress score:** "
                f"{heat}/100 "
                f"({severity(heat)}). "
                "Monitor crop temperature and water availability."
            )


        elif (
            "waterlogging" in q
            or "standing water" in q
        ):

            state = waterlog_status(
                waterlog
            )

            answer = (
                f"**Waterlogging:** "
                f"{state[0]}. "
                f"{state[1]}"
            )


        elif "rain" in q:

            state = rain_status(
                rain
            )

            answer = (
                f"**Rain status:** "
                f"{state[0]}. "
                f"{state[1]}"
            )


        elif "pump" in q:

            fault, reason = (
                pump_fault_status(
                    live["pump"],
                    live["flow"]
                )
            )

            answer = (
                f"**Pump health:** "
                f"{fault}. {reason}"
            )


        elif (
            "disease" in q
            or "leaf" in q
        ):

            if disease_label:

                readable = clean_label(
                    disease_label
                )

                answer = (
                    f"**Leaf disease prediction:** "
                    f"{readable} "
                    f"({disease_conf:.1f}% confidence). "
                    f"{disease_direction(readable, disease_conf)}"
                )

            elif green_gram_detected:

                answer = (
                    f"**Green Gram screening:** "
                    f"{green_gram_status}. "
                    f"{green_gram_recommendation}"
                )

            else:

                answer = (
                    "Upload a clear crop or leaf image "
                    "to perform the disease prediction."
                )


        elif "pest" in q:

            if pests:

                answer = (
                    f"Possible pest detected: "
                    f"{pests[0][0]} "
                    f"({pests[0][1]:.1f}%). "
                    "Verify visually before taking control action."
                )

            else:

                answer = (
                    "No pest was detected at the current model threshold."
                )


        elif (
            "nutrient" in q
            or "deficien" in q
            or "fertilizer" in q
        ):

            answer = (
                f"**Visual nutrient-stress score:** "
                f"{nutrient_score}/100. "
                f"{nutrient_pattern}"
            )


        elif (
            "drought" in q
            or "water stress" in q
        ):

            answer = (
                f"**Drought / water-stress score:** "
                f"{drought}/100 "
                f"({severity(drought)})."
            )


        elif (
            "risk" in q
            or "overall condition" in q
        ):

            answer = (
                f"**Agricultural Risk Score:** "
                f"{combined_risk}/100 "
                f"({severity(combined_risk)}). "
                "This combines environmental, water and crop-health signals."
            )


        elif (
            "what should i do" in q
            or "what can i do" in q
            or "what do i do" in q
        ):

            answer = (
                "**Recommended direction:** "
                +
                recommendations[0]
            )


        else:

            answer = (
                "I can help with irrigation, soil moisture, "
                "water availability, rain, waterlogging, pump health, "
                "leaf disease, pests, nutrient stress, drought, heat "
                "and overall agricultural risk."
            )


        st.success(
            "🌱 Farmer Assistant"
        )

        st.write(
            answer
        )


# ============================================================
# SMART FARMING CAPABILITIES
# ============================================================

st.markdown(
    '<div class="section-heading">🌾 Smart Farming Capabilities</div>',
    unsafe_allow_html=True
)

capabilities = [

    "📡 Multi-sensor ESP32 monitoring",
    "💧 Intelligent irrigation",
    "🌧️ Rain protection",
    "🌊 Waterlogging protection",
    "🚰 Flow-based pump fault detection",
    "📱 Blynk IoT monitoring",
    "🌱 Green Gram AI-assisted health screening",
    "🦠 Single-best leaf disease prediction",
    "🐛 Pest detection",
    "🌿 Nutrient-stress assessment",
    "🏜️ Drought / water-stress scoring",
    "🌡️ Heat-stress scoring",
    "📊 Agricultural Risk Score",
    "👨‍🌾 Farmer recommendation engine",
    "💬 Farmer Help Assistant",

]


cap_cols = st.columns(
    4
)


for index, capability in enumerate(
    capabilities
):

    with cap_cols[
        index % 4
    ]:

        st.success(
            capability
        )


# ============================================================
# INTEGRATED SMART FARMING ARCHITECTURE
# ============================================================

st.markdown(
    '<div class="section-heading">⚙️ Integrated Smart Farming Architecture</div>',
    unsafe_allow_html=True
)

st.info(
    "📡 ESP32 Sensors → "
    "☁️ Blynk IoT → "
    "📊 Live Farm State → "
    "🧠 Intelligent Decision Engine → "
    "💧 Irrigation & Protection → "
    "🔬 AI Leaf Analysis → "
    "🌱 Green Gram Health Screening → "
    "🐛 Pest Analysis → "
    "🌿 Nutrient Screening → "
    "🏜️ Drought + 🌡️ Heat Analysis → "
    "🚨 Agricultural Risk Score → "
    "👨‍🌾 Farmer Recommendation → "
    "💬 Farmer Help Assistant"
)


# ============================================================
# AGRICULTURAL SAFETY NOTE
# ============================================================

st.markdown(
    """
    <div class="warning-note">

    <b>🛡️ Important Agricultural Safety Note</b><br><br>

    AI disease and pest outputs are decision-support results.
    Green Gram visual health screening is an AI-assisted screening
    method based on visible image characteristics and does not provide
    definitive disease diagnosis.

    Visual nutrient assessment is only a screening method and does not
    directly measure soil nutrient concentration.

    Disease, pest and nutrient conditions should be confirmed through
    field inspection and appropriate agricultural testing before
    treatment or fertilizer decisions.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        🌱 Smart Plant Guardian
        <br>

        AI + IoT Field-Deployable Smart Farming Assistant
        <br>

        🚜 Monitor • Analyze • Decide • Act • Protect

    </div>
    """,
    unsafe_allow_html=True
)
