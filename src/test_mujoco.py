from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("MuJoCo loaded successfully!")
print("Number of joints:", model.njnt)
print("Number of actuators:", model.nu)
print("Simulation timestep:", model.opt.timestep)
