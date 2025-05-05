import gymnasium as gym
import mani_skill2_real2sim.envs

ENVIRONMENTS = [
    "google_robot_pick_coke_can",
    "google_robot_pick_horizontal_coke_can",
    "google_robot_pick_vertical_coke_can",
    "google_robot_pick_standing_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near_v0",
    "google_robot_move_near_v1",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_open_top_drawer",
    "google_robot_open_middle_drawer",
    "google_robot_open_bottom_drawer",
    "google_robot_close_drawer",
    "google_robot_close_top_drawer",
    "google_robot_close_middle_drawer",
    "google_robot_close_bottom_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_place_in_closed_top_drawer",
    "google_robot_place_in_closed_middle_drawer",
    "google_robot_place_in_closed_bottom_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
    # * new task: unseen target, unseen action combinations
    "widowx_carrot_on_coke_can",
    "widowx_carrot_on_green_cube",
    "widowx_plate_on_green_cube",
    "widowx_coke_can_on_pepsi_can",
    # * generatization test
    "widowx_cube_on_plate_clean",
    "widowx_small_plate_on_green_cube_clean",
    "widowx_coke_can_on_plate_clean",
    "widowx_carrot_on_Sponge_clean",
    "widowx_carrot_on_keyboard_clean",
    "widowx_coke_can_on_keyboard_clean",
    # * object distraction
    "widowx_spoon_on_towel_distract",
    "widowx_carrot_on_plate_distract",
    "widowx_carrot_on_keyboard_distract",
    "widowx_coke_can_on_plate_distract",
    "widowx_coke_can_on_keyboard_distract",
    # * language variation
    "widowx_carrot_on_plate_lang_common", # rabbit, no distract
    "widowx_carrot_on_plate_lang_action",
    "widowx_carrot_on_plate_lang_neg",
    "widowx_carrot_on_plate_lang_neg_action", # on the table not on the plate
    "widowx_carrot_on_plate_lang_common_distract", # rabbit
    "widowx_spoon_on_towel_lang_common",
    "widowx_spoon_on_towel_lang_common_distract",
    "widowx_eggplant_in_basket_lang_color",
    "widowx_eggplant_in_basket_lang_common",
    "widowx_carrot_on_keyboard_lang_common",
    "widowx_coke_can_on_plate_lang_neg",
    "widowx_coke_can_on_plate_lang_common_distract", # thirsty
]

ENVIRONMENT_MAP = {
    "google_robot_pick_coke_can": ("GraspSingleOpenedCokeCanInScene-v0", {}),
    "google_robot_pick_horizontal_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"lr_switch": True},
    ),
    "google_robot_pick_vertical_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"laid_vertically": True},
    ),
    "google_robot_pick_standing_coke_can": (
        "GraspSingleOpenedCokeCanInScene-v0",
        {"upright": True},
    ),
    "google_robot_pick_object": ("GraspSingleRandomObjectInScene-v0", {}),
    "google_robot_move_near": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_move_near_v0": ("MoveNearGoogleBakedTexInScene-v0", {}),
    "google_robot_move_near_v1": ("MoveNearGoogleBakedTexInScene-v1", {}),
    "google_robot_open_drawer": ("OpenDrawerCustomInScene-v0", {}),
    "google_robot_open_top_drawer": ("OpenTopDrawerCustomInScene-v0", {}),
    "google_robot_open_middle_drawer": ("OpenMiddleDrawerCustomInScene-v0", {}),
    "google_robot_open_bottom_drawer": ("OpenBottomDrawerCustomInScene-v0", {}),
    "google_robot_close_drawer": ("CloseDrawerCustomInScene-v0", {}),
    "google_robot_close_top_drawer": ("CloseTopDrawerCustomInScene-v0", {}),
    "google_robot_close_middle_drawer": ("CloseMiddleDrawerCustomInScene-v0", {}),
    "google_robot_close_bottom_drawer": ("CloseBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_drawer": ("PlaceIntoClosedDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_top_drawer": ("PlaceIntoClosedTopDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_middle_drawer": ("PlaceIntoClosedMiddleDrawerCustomInScene-v0", {}),
    "google_robot_place_in_closed_bottom_drawer": ("PlaceIntoClosedBottomDrawerCustomInScene-v0", {}),
    "google_robot_place_apple_in_closed_top_drawer": (
        "PlaceIntoClosedTopDrawerCustomInScene-v0", 
        {"model_ids": "baked_apple_v2"}
    ),
    "widowx_spoon_on_towel": ("PutSpoonOnTableClothInScene-v0", {}),
    "widowx_carrot_on_plate": ("PutCarrotOnPlateInScene-v0", {}),
    "widowx_stack_cube": ("StackGreenCubeOnYellowCubeBakedTexInScene-v0", {}),
    "widowx_put_eggplant_in_basket": ("PutEggplantInBasketScene-v0", {}),
    # * new task: unseen target, unseen action combinations, too hard
    "widowx_carrot_on_coke_can": ("PutCarrotOnCokeCanInScene-v1", {}),
    "widowx_carrot_on_green_cube": ("PutCarrotOnGreenCubeInScene-v1", {}),
    "widowx_plate_on_green_cube": ("PutPlateOnGreenCubeInScene-v1", {}),
    "widowx_coke_can_on_pepsi_can": ("PutCokeCanOnPepsiCanInScene-v1", {}),
    # * generatization test
    "widowx_cube_on_plate_clean": ("PutGreenCubeOnPlateInScene-v2", {}),
    "widowx_small_plate_on_green_cube_clean": ("PutSmallPlateOnGreenCubeInScene-v2", {}),
    "widowx_coke_can_on_plate_clean": ("PutCokeCanOnPlateInScene-v2", {}),
    "widowx_carrot_on_Sponge_clean": ("PutCarrotOnSpongeInScene-v2", {}),
    "widowx_carrot_on_keyboard_clean": ("PutCarrotOnKeyboardInScene-v2", {}),
    "widowx_coke_can_on_keyboard_clean": ("PutCokeCanOnKeyboardInScene-v2", {}),
    # * object distraction
    "widowx_spoon_on_towel_distract": ("PutSpoonOnTableClothInScene-distract", {}),
    "widowx_carrot_on_plate_distract": ("PutCarrotOnPlateInScene-distract", {}),
    "widowx_carrot_on_keyboard_distract": ("PutCarrotOnKeyboardInScene-distract", {}),
    "widowx_coke_can_on_plate_distract": ("PutCokeCanOnPlateInScene-distract", {}),
    "widowx_coke_can_on_keyboard_distract": ("PutCokeCanOnKeyboardInScene-distract", {}),
    # * language variation
    "widowx_carrot_on_plate_lang_common": ("PutCarrotOnPlateInScene-LangV1", {}), # rabbit, no distract
    "widowx_carrot_on_plate_lang_action": ("PutCarrotOnPlateInScene-LangV2", {}), # pick up the carrot and drop it off on the plate
    "widowx_carrot_on_plate_lang_neg": ("PutCarrotOnPlateInScene-LangV3", {}), # put the carrot on the plate, not the towel
    "widowx_carrot_on_plate_lang_neg_action": ("PutCarrotOnPlateInScene-LangV4", {}), # on the table not on the plate
    "widowx_carrot_on_plate_lang_common_distract": ("PutCarrotOnPlateInScene-LangV5", {}), # rabbit
    "widowx_spoon_on_towel_lang_common": ("PutSpoonOnTableClothInScene-LangV1", {}),
    "widowx_spoon_on_towel_lang_common_distract": ("PutSpoonOnTableClothInScene-LangV2", {}),
    "widowx_eggplant_in_basket_lang_color": ("PutEggplantInBasketScene-LangV1", {}),
    "widowx_eggplant_in_basket_lang_common": ("PutEggplantInBasketScene-LangV2", {}),
    "widowx_carrot_on_keyboard_lang_common": ("PutCarrotOnKeyboardInScene-LangV1", {}),
    "widowx_coke_can_on_plate_lang_neg": ("PutCokeCanOnPlateInScene-LangV1", {}),
    "widowx_coke_can_on_plate_lang_common_distract": ("PutCokeCanOnPlateInScene-LangV2", {}), # thirsty
}


def make(task_name):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env
