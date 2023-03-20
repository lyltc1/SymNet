OUTPUT_ROOT = "output"
OUTPUT_DIR = "auto"  # automatic set to model_name+dataset_name+obj_id+time
RESUME = None

MODEL = dict(
    NAME="SymNet",
    BACKBONE=dict(
        ARCH="resnet",
        NUM_LAYERS=34,
        INPUT_CHANNEL=3,
        CONCAT=True,
        FREEZE=False,
        # PRETRAINED="torchvision://resnet34",  # fixed
        PRETRAINED="pretrained_backbone/resnet34-333f7ec4.pth",
    ),
    GEOMETRY_NET=dict(
        ARCH="aspp",  # choose from ["aspp", "cdpn"]
        FREEZE=False,
        VISIB_MASK_LOSS_TYPE="L1",  # choose from ["BCE", "L1"]
        VISIB_MASK_LW=1,
        AMODAL_MASK_LOSS_TYPE="L1",  # choose from ["BCE", "L1"]
        AMODAL_MASK_LW=1,
        CODE_LOSS_TYPE="BCE",  # choose from ["BCE", "L1"]
        CODE_LW=3,
    ),
    PNP_NET=dict(
        ARCH="ConvPnPNet",    # choose from ["ConvPnPNet"]
        FREEZE=False,
        LR_MULT=1.0,
        # output type
        R_type='R_allo_6d',  # choose from ['R_allo_6d', 'R_allo']
        # R_allo_sym loss, only use in sym_continuous
        R_ALLO_SYM_LW=0.0,
        R_ALLO_SYM_LOSS_TYPE="L1",
        # SITE loss
        t_type='SITE',
        SITE_XY_LW=1.0,
        SITE_XY_LOSS_TYPE="L1",  # choose from ["L1", "MSE"]
        SITE_Z_LW=0.001,  # because the unit is mm
        SITE_Z_LOSS_TYPE="L1",  # choose from ["L1", "MSE"]
        # point matching loss
        PM_LW=1.0,
        PM_R_ONLY=True,
        PM_LOSS_TYPE="L1",  # choose from ["L1", "L2", "MSE", "SMOOTH_L1"]
        PM_NORM_BY_EXTENT=True,
        PM_NUM_POINTS=3000,
        PM_LOSS_SYM=True,
    )
)

SOLVER = dict(
    OPTIMIZER_CFG=dict(type="Ranger", lr=1e-4, weight_decay=0),
    LR_SCHEDULER_CFG=dict(type="LambdaLR", warm=1000),
    # LR_SCHEDULER_CFG choose from ["LambdaLR", "MultiStepLR", "CosineAnnealingLR"],
    # need the scheduler's special params, e.g. 'milestones' for "MultiStepLR"
)

DATASETS = dict(
    NAME="lmo",
    TRAIN=("train_real", "train_pbr"),  # tuple, which is the subset of ("pbr", "real",)
    TEST=("test",),
    TEST_KEYFRAME="test_targets_bop19.json",  # choose from [None, "test_targets_bop19.json",
                                              #              "ycbv_test_targets_keyframe.json"]
    TEST_DETECTION_TYPE="type1",  # choose from ["type1" for yolox, "type2" for maskrcnn, "type3"]
    TEST_DETECTION_PATH='zebrapose_detections/lmo/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_lmo_pbr.json',
    TEST_SCORE_THR=0.5,
    TEST_TOP_K_PER_OBJ=1,
    RES_CROP=256,
    OBJ_IDS=None,  # OBJ_IDS=[15, 18],  # should be consistent with NUM_CLASSES
    NUM_CLASSES=None,  # NUM_CLASSES=2,  # should be consistent with OBJ_IDS
    COLOR_AUG_PROB=0.8,
    OCCLUDE_AUG_PROB=0.0,
    BG_AUG_PROB=0.5,
    BG_AUG_TYPE="VOC",  # choose from ["VOC", "VOC_table"]
    SYM_OBJS_ID="bop",  # choose from ["bop", or [13, 16, 19, 20, 21]
)

TRAIN = dict(
    PRINT_FREQ=10,
    NUM_WORKERS=2,
    BATCH_SIZE=32,  # BATCH_SIZE for one gpu
    TOTAL_EPOCHS=600,
    DEBUG_MODE=False,  # visualize some images during training and more out_dict
)

TEST = dict(
    EVAL_PERIOD=0,
    EVALUAGTE_TYPE="custom",   # choose from ["bop", "custom"]
    AMP_TEST=False,  # TODO, what's the use
    USE_PNP=False,
)

VAL = dict(
    EVAL_CACHED=False,  # if the predicted poses have been saved
    EVAL_PRINT_ONLY=False,  # if the predicted poses have been saved
    SPLIT_TYPE="",
)
