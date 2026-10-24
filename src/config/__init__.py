"""Validated configuration; numerical controls never come from an LLM."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LLMConfig(Section):
    mode: Literal["offline", "auto", "required"] = "auto"
    model: str = "qwen3-vl-plus"
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_s: float = Field(30, gt=0)
    max_calls: int = Field(24, ge=1)


class CameraConfig(Section):
    width: int = Field(640, ge=64)
    height: int = Field(480, ge=48)
    detail_width: int = Field(1280, ge=640)
    detail_height: int = Field(960, ge=480)
    fovy_deg: float = Field(65, gt=5, lt=150)
    mode: Literal["gantry", "wrist"] = "wrist"
    # Calibration/scan poses, never object positions.
    viewpoints: list[tuple[float, float, float]] = [(0.85, -0.85, 1.35), (0.0, 0.0, 1.65), (-0.85, 0.85, 1.35)]
    look_at: tuple[float, float, float] = (0, 0, 0.06)
    view_targets: list[tuple[float,float,float]] = []
    wrist_scan_angles: list[float] = [-1.57, -0.7, 0.1, 0.9, 1.7, 2.5]


class PerceptionConfig(Section):
    detector: Literal["fallback", "grounding_dino", "vlm"] = "fallback"
    segmenter: Literal["color", "sam2"] = "color"
    dino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam_checkpoint: str = ""
    sam_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml"
    confidence_threshold: float = Field(.60, gt=0, le=1)
    min_pixels: int = Field(30, ge=3)
    min_depth_m: float = .025
    max_depth_m: float = 3.0
    voxel_m: float = .004
    association_m: float = .10
    memory_ttl_s: float = 40
    max_uncertainty_m: float = .025


class PlannerConfig(Section):
    cartesian_step_m: float = Field(.025, gt=0, le=.05)
    joint_step_rad: float = Field(.06, gt=0, le=.15)
    rrt_timeout_s: float = Field(2, gt=0)
    approach_clearance_m: float = .12
    side_approach_clearance_m: float = Field(.06,gt=.03,le=.15)
    transport_clearance_m: float = .13
    side_transport_clearance_m: float = Field(.08,gt=.03,le=.20)
    palm_to_contact_m: float = .147
    max_gripper_width_m: float = .08
    push_tool_radius_m: float = Field(.014,gt=0,le=.04)
    release_clearance_m: float = Field(.008,gt=0,le=.01)
    grasp_insertion_m: float = .035
    adjacent_gap_m: float = .025
    base_clearance_radius_m: float = .28
    transfer_radius_m: float = .43
    side_transfer_radius_m: float = Field(.48,gt=.28,le=.65)


class SafetyConfig(Section):
    workspace: tuple[tuple[float, float], ...] = ((-.75, .75), (-.75, .75), (0, 1.1))
    max_velocity: float = Field(.45, gt=0, le=1.6)
    carrying_velocity: float = Field(.20, gt=0, le=.45)
    joint_tracking_tolerance_rad: float = .12
    contact_force_limit_n: float = 20
    min_grasp_force_n: float = .35
    slip_distance_m: float = .018
    placement_xy_tolerance_m: float = .09
    placement_z_tolerance_m: float = .025
    stable_displacement_m: float = .01
    collision_margin_m: float = .008


class RecoveryConfig(Section):
    max_grasp_attempts: int = Field(4, ge=1, le=8)
    max_recoveries: int = Field(6, ge=0, le=20)
    max_steps: int = Field(70, ge=1, le=300)
    timeout_s: float = Field(450, gt=0)
    llm_after_failures: int = Field(2, ge=1)


class RobotConfig(Section):
    backend: Literal["mujoco"] = "mujoco"
    gripper_force_target_n: float = 2.0
    gripper_squeeze_m: float = .006
    physics_settle_s: float = .2


class LocalizationConfig(Section):
    enabled: bool = True
    grounding_mode: Literal["auto", "vlm", "local"] = "auto"
    max_vlm_calls: int = Field(10, ge=0, le=30)
    detail_vlm_reserve: int = Field(2, ge=0, le=10)
    max_search_views: int = Field(6, ge=1, le=12)
    max_local_observations: int = Field(4, ge=2, le=10)
    max_corrections: int = Field(3, ge=1, le=8)
    precise_uncertainty_m: float = Field(.012, gt=0, le=.025)
    precise_max_range_m: float = Field(.50, gt=.10, le=1.)
    min_valid_depth_ratio: float = Field(.70, gt=0, le=1)
    max_lock_age_s: float = Field(3., gt=0)
    observation_distance_m: float = Field(.38, gt=.20, le=.60)
    correction_step_m: float = Field(.012, gt=0, le=.025)
    correction_tolerance_m: float = Field(.004, gt=0, le=.015)
    tracking_iou_min: float = Field(.15, ge=0, le=1)
    reground_interval_s: float = Field(3.,gt=0)


class AdaptiveGraspConfig(Section):
    fast_top_score_threshold: float = Field(.80, ge=0, le=1)
    complexity_threshold: float = Field(.55, ge=0, le=1)
    max_multiview_observations: int = Field(3, ge=1, le=6)
    min_geometry_completeness: float = Field(.65, gt=0, le=1)
    profile_slices: int = Field(7, ge=3, le=20)
    min_slice_points: int = Field(12, ge=3)
    finger_clearance_m: float = Field(.008, ge=0)
    max_cloud_points: int = Field(12000, ge=100)
    llm_reasoner_enabled: bool = True
    reasoner_candidate_margin: float = Field(.10, ge=0, le=1)
    max_reasoner_calls: int = Field(2, ge=0, le=6)
    side_grasp_force_target_n: float = Field(3.,gt=0,le=10)


class SpatialConfig(Section):
    default_reference_frame: Literal["operator", "robot_base", "world", "object"] = "operator"
    default_distance_semantics: Literal["edge_gap", "center_distance", "surface_offset"] = "edge_gap"
    operator_origin: tuple[float,float,float] = (0.,0.,0.)
    operator_yaw_deg: float = 0.
    default_tolerance_m: float = Field(.01, gt=0, le=.05)
    direction_alignment_tolerance_m: float = Field(.025, gt=0)
    near_min_gap_m: float = Field(.05, ge=0)
    near_max_gap_m: float = Field(.12, gt=0)
    candidate_step_m: float = Field(.02, gt=.005, le=.05)
    min_table_edge_margin_m: float = Field(.025, ge=0)
    min_obstacle_margin_m: float = Field(.012, ge=0)


class PlacementCorrectionConfig(Section):
    enable_push: bool = True
    max_push_distance_m: float = Field(.05, gt=0, le=.08)
    push_step_m: float = Field(.015, gt=0, le=.02)
    max_push_attempts: int = Field(3, ge=1, le=6)
    max_regrasps: int = Field(2, ge=0, le=4)
    max_height_width_ratio: float = Field(1.8, gt=0)
    min_table_edge_margin_m: float = Field(.04, ge=.02)
    push_velocity_m_s: float = Field(.03, gt=0, le=.06)
    push_force_limit_n: float = Field(6., gt=0, le=20)
    min_observed_motion_m: float = Field(.001, gt=0)


class AgentConfig(Section):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)
    localization: LocalizationConfig = Field(default_factory=LocalizationConfig)
    grasp: AdaptiveGraspConfig = Field(default_factory=AdaptiveGraspConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    placement_correction: PlacementCorrectionConfig = Field(default_factory=PlacementCorrectionConfig)
    log_root: str = "outputs/agent"

    @classmethod
    def load(cls, path: str | Path | None = None):
        return cls() if path is None else cls.model_validate_json(Path(path).read_text())
