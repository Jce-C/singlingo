"""
models.py — Pydantic schemas para request/response de la API.

Soporta:
  - Dos manos (landmarks_right + landmarks_left)
  - World landmarks 3D (para features v6 rotation-invariant)
  - Porcentajes de flexión por dedo
"""
from __future__ import annotations

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    x: float = Field(..., ge=-5.0, le=5.0)
    y: float = Field(..., ge=-5.0, le=5.0)
    z: float = Field(default=0.0, ge=-5.0, le=5.0)


class PoseLandmarkPoint(BaseModel):
    x:          float = Field(..., ge=-2.0, le=2.0)
    y:          float = Field(..., ge=-2.0, le=2.0)
    z:          float = Field(default=0.0, ge=-2.0, le=2.0)
    visibility: float = Field(default=0.0, ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    landmarks: Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)
    landmarks_right:       Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)
    landmarks_left:        Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)
    world_landmarks_right: Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21,
                                                                  description="World landmarks 3D (metros) mano derecha")
    world_landmarks_left:  Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21,
                                                                  description="World landmarks 3D (metros) mano izquierda")
    pose_landmarks: Optional[List[PoseLandmarkPoint]] = Field(default=None, min_length=33, max_length=33,
                                                               description="33 landmarks de cuerpo de PoseLandmarker")
    country:        str            = Field(default="asl")
    target_sign_id: Optional[str] = Field(default=None)


class PredictResponse(BaseModel):
    sign_id:            str
    sign_name:          str
    confidence:         float         = Field(..., ge=0.0, le=1.0)
    fingers_up:         List[bool]
    fingers_up_left:    Optional[List[bool]]  = None
    flexion_pcts:       List[float]           = Field(default_factory=lambda: [0.0]*5,
                                                      description="% flexión por dedo [0=extendido, 1=doblado]")
    flexion_pcts_left:  Optional[List[float]] = None
    tips:               List[str]    = []
    is_correct:         Optional[bool] = None
    score:              float        = Field(default=0.0, ge=0.0, le=100.0)
    two_handed:         bool         = False
    hands_detected:     int          = 1


class FeatureTemplate(BaseModel):
    thumb_ext:   float
    index_ext:   float
    middle_ext:  float
    ring_ext:    float
    pinky_ext:   float
    pinch_dist:  float
    spread:      float
    thumb_horiz: float


class SignData(BaseModel):
    id:                  str
    name:                str
    emoji:               str  = "🤟"
    category:            str
    difficulty:          int  = Field(..., ge=1, le=5)
    description:         str
    tips:                List[str]
    finger_states:       List[bool]          = Field(..., min_length=5, max_length=5)
    finger_states_other: Optional[List[bool]] = None
    feature_template:    Optional[Dict[str, float]] = None
    country:             str  = "asl"
    two_handed:          bool = False
    body_zone:           Optional[str]  = None   # "face"|"chin"|"chest"|"belly"|"anywhere"
    requires_pose:       bool           = False   # True si la seña necesita PoseLandmarker


class SignsResponse(BaseModel):
    country: str
    total:   int
    signs:   List[SignData]


class ProgressEntry(BaseModel):
    sign_id:         str
    country:         str          = "asl"
    success:         bool
    response_time_ms: Optional[int] = None


class SignProgress(BaseModel):
    sign_id:     str
    sign_name:   str
    attempts:    int
    successes:   int
    accuracy:    float
    best_time_ms: Optional[int]
    mastered:    bool


class ProgressSummary(BaseModel):
    total_attempts:   int
    total_successes:  int
    overall_accuracy: float
    current_streak:   int
    best_streak:      int
    signs_mastered:   int
    by_sign:          List[SignProgress]
    by_country:       Dict[str, int]


class HealthResponse(BaseModel):
    status:               str
    model_loaded:         Dict[str, bool]
    supported_countries:  List[str]
    total_signs:          int


class CollectSample(BaseModel):
    landmarks:             List[LandmarkPoint] = Field(..., min_length=21, max_length=21)
    world_landmarks:       Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)
    landmarks_left:        Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)
    world_landmarks_left:  Optional[List[LandmarkPoint]] = Field(default=None, min_length=21, max_length=21)


class CollectRequest(BaseModel):
    sign_id: str   = Field(..., description="ID de la seña (e.g. 'A', 'HOLA')")
    country: str   = Field(default="lsc")
    samples: List[CollectSample] = Field(..., min_length=1)


class CollectResponse(BaseModel):
    sign_id:       str
    country:       str
    saved:         int
    total_for_sign: int
    message:       str


class SignSampleCount(BaseModel):
    sign_id: str
    count:   int
    enough:  bool


class TrainStatusResponse(BaseModel):
    country:          str
    total_signs:      int
    signs_with_data:  int
    signs_ready:      int
    ready_to_train:   bool
    samples_per_sign: List[SignSampleCount]
    min_samples_needed: int


class RetrainResponse(BaseModel):
    success:       bool
    country:       str
    message:       str
    accuracy:      Optional[float] = None
    total_samples: Optional[int]   = None


class CNNPredictRequest(BaseModel):
    image_b64: str = Field(..., description="PNG/JPEG 100×100 en base64")
    fmt:       str = Field(default="png")


class CNNTopK(BaseModel):
    letter:     str
    confidence: float


class CNNPredictResponse(BaseModel):
    letter:      str
    confidence:  float
    top3:        List[CNNTopK]
    model_ready: bool
