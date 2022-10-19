SMPL_FLDR = 'dataset/body_models/smpl/'
SMPL_REGRESSOR = {'J17': 'dataset/body_models/J_regressor_h36m.npy',
                  'OP19': 'dataset/body_models/SMPLCOCORegressor.npy',
                  'TC16': 'dataset/body_models/SMPLTotalCapture_male.npy'}
PELVIS_IDX = {'J17': 0, 'OP19': 13, 'TC16': 0}

SMPL_MEAN_PARAMS = 'dataset/body_models/smpl_mean_params2.npz'
AMASS_TRAIN_LABEL_PTH = 'dataset/amass_db_train.pt'
AMASS_TEST_LABEL_PTH = 'dataset/amass_db_test.pt'
AMASS_FPS = 25
CALIB_FILE = 'dataset/basic_calib.npy'

SMPL_JOINT_NAMES = [
    'Pelvis', 'Left Hip', 'Right Hip', 'Spine 1 (Lower)', 'Left Knee',
    'Right Knee', 'Spine 2 (Middle)', 'Left Ankle', 'Right Ankle',
    'Spine 3 (Upper)', 'Left Foot', 'Right Foot', 'Neck',
    'Left Shoulder (Inner)', 'Right Shoulder (Inner)', 'Head',
    'Left Shoulder (Outer)', 'Right Shoulder (Outer)', 'Left Elbow',
    'Right Elbow', 'Left Wrist', 'Right Wrist', 'Left Hand', 'Right Hand']

SENSOR_TO_SMPL_MAP = {
    'Head': 'Head', 'Sternum': 'Spine 3 (Upper)', 'Pelvis': 'Pelvis',
    'L_UpArm': 'Left Shoulder (Outer)', 'R_UpArm': 'Right Shoulder (Outer)',
    'L_LowArm': 'Left Elbow', 'R_LowArm': 'Right Elbow',
    'L_UpLeg': 'Left Hip', 'R_UpLeg': 'Right Hip',
    'L_LowLeg': 'Left Knee', 'R_LowLeg': 'Right Knee',
    'L_Foot': 'Left Ankle', 'R_Foot': 'Right Ankle'}

SENSOR_LIST = [
    'Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm',
    'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg',
    'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot',
]

SENSOR_TO_VERTS = {
    'Head': 411, 'Sternum': 3076, 'Pelvis': 3502, 'L_UpArm': 1379,
    'R_UpArm': 4849, 'L_LowArm': 1952, 'R_LowArm': 5422, 'L_UpLeg': 847,
    'R_UpLeg': 4712, 'L_LowLeg': 1373, 'R_LowLeg': 4561, 'L_Foot': 3345,
    'R_Foot': 6745
}
