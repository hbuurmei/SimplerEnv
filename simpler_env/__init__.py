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
    # * added 05-08
    "widowx_coke_can_on_plate_lang_common",
    "widowx_carrot_on_sponge_larger",
    "widowx_eggplant_on_sponge",
    "widowx_eggplant_on_sponge_larger",
    "widowx_eggplant_in_basket_lang_action",
    "widowx_spoon_on_towel_lang_action",
    "widowx_stack_cube_lang_action",
    "widowx_pepsi_on_plate_clean",
    # lighting
    "widowx_carrot_on_plate_brighter",
    "widowx_carrot_on_plate_darker",
    "widowx_eggplant_in_basket_brighter",
    "widowx_eggplant_in_basket_darker",
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
    # * generatization test, clean background
    "widowx_cube_on_plate_clean": ("PutGreenCubeOnPlateInScene-v2", {}), # seen source and target, unseen combination
    "widowx_small_plate_on_green_cube_clean": ("PutSmallPlateOnGreenCubeInScene-v2", {}), # seen source and target, unseen combination (a reverse)
    "widowx_coke_can_on_plate_clean": ("PutCokeCanOnPlateInScene-v2", {}), # ood source
    "widowx_carrot_on_Sponge_clean": ("PutCarrotOnSpongeInScene-v2", {}), # seen source and target, unseen combination
    "widowx_carrot_on_keyboard_clean": ("PutCarrotOnKeyboardInScene-v2", {}), # ood target
    "widowx_coke_can_on_keyboard_clean": ("PutCokeCanOnKeyboardInScene-v2", {}), # ood source and ood target
    # * object distraction
    "widowx_spoon_on_towel_distract": ("PutSpoonOnTableClothInScene-distract", {}), # spoon on towel, + plate + carrot, [original, source distract + target distract]
    "widowx_carrot_on_plate_distract": ("PutCarrotOnPlateInScene-distract", {}), # + spoon + towel [original, source distract + target distract]
    "widowx_carrot_on_keyboard_distract": ("PutCarrotOnKeyboardInScene-distract", {}), # + plate + spoon [ood target, source + target distract]
    "widowx_coke_can_on_plate_distract": ("PutCokeCanOnPlateInScene-distract", {}), # + pepsi can + carrot [ood source, source distract]
    "widowx_coke_can_on_keyboard_distract": ("PutCokeCanOnKeyboardInScene-distract", {}), # + plate + carrot [ood source ood target, source + target distract]
    # * language variation
    "widowx_carrot_on_plate_lang_common": ("PutCarrotOnPlateInScene-LangV1", {}), # carrot -> rabbits' favorite vegetable, no distract
    "widowx_carrot_on_plate_lang_action": ("PutCarrotOnPlateInScene-LangV2", {}), # pick up the carrot and drop it off on the plate
    "widowx_carrot_on_plate_lang_neg": ("PutCarrotOnPlateInScene-LangV3", {}), # put the carrot on the plate, not the towel
    "widowx_carrot_on_plate_lang_neg_action": ("PutCarrotOnPlateInScene-LangV4", {}), # on the table not on the plate
    "widowx_carrot_on_plate_lang_common_distract": ("PutCarrotOnPlateInScene-LangV5", {}), # rabbit, + distract rabbit + eggplant
    "widowx_spoon_on_towel_lang_common": ("PutSpoonOnTableClothInScene-LangV1", {}), # spoon -> kitchenware for eating soup
    "widowx_spoon_on_towel_lang_common_distract": ("PutSpoonOnTableClothInScene-LangV2", {}), # spoon -> kitchenware for eating soup, + sponge + eggplant
    "widowx_eggplant_in_basket_lang_color": ("PutEggplantInBasketScene-LangV1", {}), # eggplant -> purple object
    "widowx_eggplant_in_basket_lang_common": ("PutEggplantInBasketScene-LangV2", {}), # yellow basket -> where the dishes usually get dried
    "widowx_carrot_on_keyboard_lang_common": ("PutCarrotOnKeyboardInScene-LangV1", {}), # keyboard -> tool for typing words
    "widowx_coke_can_on_plate_lang_neg": ("PutCokeCanOnPlateInScene-LangV1", {}), # "put coke can, not the carrot, not the pepsi can, on the plate"
    "widowx_coke_can_on_plate_lang_common_distract": ("PutCokeCanOnPlateInScene-LangV2", {}), # thirsty "put the object that one needs the most when they are thirsty on plate" + carrot + eggplant
    # * added 05-08
    "widowx_coke_can_on_plate_lang_common": ("PutCokeCanOnPlateInScene-LangV3", {}), # languge commonsense, no distract, (thirsty)
    "widowx_carrot_on_sponge_larger": ("PutCarrotOnSpongeLargerInScene-v2", {}), # TODO: sponge: see if grasp correct is severely affected by unusual target
    "widowx_eggplant_on_sponge": ("PutEggplantOnSpongeInScene-v2", {}),
    "widowx_eggplant_on_sponge_larger": ("PutEggplantOnSpongeLargerInScene-v2", {}),
    "widowx_eggplant_in_basket_lang_action": ("PutEggplantInBasketScene-LangV3", {}), # TODO: language action, see if performance also severely drop like the carrot case
    "widowx_spoon_on_towel_lang_action": ("PutSpoonOnTableClothInScene-LangV3", {}), # language action
    "widowx_stack_cube_lang_action": ("StackGreenCubeOnYellowCubeBakedTexInScene-LangV1", {}),
    "widowx_pepsi_on_plate_clean": ("PutPepsiCanOnPlateInScene-v2", {}), # TODO: another OOD source besides coke can, also a texture difference
    # lighting
    "widowx_carrot_on_plate_brighter": ("PutCarrotOnPlateInScene-light-v1", {}),
    "widowx_carrot_on_plate_darker" : ("PutCarrotOnPlateInScene-light-v2", {}),
    "widowx_eggplant_in_basket_brighter": ("PutEggplantInBasketScene-light-v1", {}),
    "widowx_eggplant_in_basket_darker": ("PutEggplantInBasketScene-light-v2", {}),
}


def make(task_name):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env
