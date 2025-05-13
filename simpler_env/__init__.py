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
    # * original tasks
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
    # * generatization test
    "widowx_cube_on_plate_clean", # seen source and target, unseen combination
    "widowx_small_plate_on_green_cube_clean", # seen source and target, unseen combination (a reverse)
    "widowx_coke_can_on_plate_clean", # ood source
    "widowx_pepsi_on_plate_clean", # another OOD source besides coke can, also a texture difference
    "widowx_carrot_on_sponge_clean", # seen source and target, unseen combination
    "widowx_eggplant_on_sponge_clean",
    "widowx_carrot_on_keyboard_clean", # ood target
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
    "widowx_carrot_on_plate_lang_common_distract", # rabbit
    "widowx_spoon_on_towel_lang_action",
    "widowx_spoon_on_towel_lang_common",
    "widowx_spoon_on_towel_lang_common_distract",
    "widowx_stack_cube_lang_action",
    "widowx_eggplant_in_basket_lang_action",
    "widowx_eggplant_in_basket_lang_color",
    "widowx_eggplant_in_basket_lang_common",
    "widowx_carrot_on_keyboard_lang_common",
    "widowx_coke_can_on_plate_lang_common",
    "widowx_coke_can_on_plate_lang_neg",
    "widowx_coke_can_on_plate_lang_common_distract", # thirsty
    
    "widowx_orange_juice_on_plate_clean", # ood source
    "widowx_orange_juice_on_plate_distract", # + carrot + orange
    "widowx_orange_juice_on_plate_lang_neg", # "put orange juice, not the orange, on the plate"
    "widowx_orange_juice_on_plate_lang_common", # "put the juice squeezed from orange on the plate"
    "widowx_orange_juice_on_plate_lang_common_distract", # "put the juice squeezed from orange on the plate" + carrot + orange
    "widowx_orange_juice_on_plate_lang_common_distractv2", # "put the drink rich in vitamin C on the plate" + coke can + orange
    "widowx_nut_on_plate_clean", # ood source
    "widowx_nut_on_plate_lang_common", # ood source + language commonsense, nut -> metal component for taming bolts
    "widowx_eggplant_on_keyboard_clean", # ood target
    "widowx_carrot_on_ramekin_clean", # ood target
    "widowx_carrot_on_wheel_clean", # ood target
    "widowx_coke_can_on_ramekin_clean", # ood source
    "widowx_coke_can_on_wheel_clean", # ood target
    "widowx_nut_on_wheel_clean", # ood target
    "widowx_cube_on_plate_lang_shape", # "put the square shaped object on the round shaped object"
    "widowx_spoon_on_towel_lang_neg", # "put the spoon on the towel, not on the plate"
    "widowx_spoon_on_towel_lang_color", # "put the shiny object with green handle on the blue object"
    "widowx_carrot_on_plate_lang_color", # "put the orange object on the yellow object"

    # # ! not in used.
    # # lighting
    # "widowx_carrot_on_plate_brighter",
    # "widowx_carrot_on_plate_darker",
    # "widowx_eggplant_in_basket_brighter",
    # "widowx_eggplant_in_basket_darker",
    # # * Not in used tasks, too difficult or difficult to evaluate
    # "widowx_carrot_on_coke_can",
    # "widowx_carrot_on_green_cube",
    # "widowx_plate_on_green_cube",
    # "widowx_coke_can_on_pepsi_can",
    # "widowx_carrot_on_Sponge_clean",
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
    # * generatization test, clean background
    "widowx_cube_on_plate_clean": ("PutGreenCubeOnPlateInScene-v2", {}), # seen source and target, unseen combination
    "widowx_small_plate_on_green_cube_clean": ("PutSmallPlateOnGreenCubeInScene-v2", {}), # seen source and target, unseen combination (a reverse)
    "widowx_coke_can_on_plate_clean": ("PutCokeCanOnPlateInScene-v2", {}), # ood source
    "widowx_pepsi_on_plate_clean": ("PutPepsiCanOnPlateInScene-v2", {}), # TODO: another OOD source besides coke can, also a texture difference
    "widowx_carrot_on_sponge_clean": ("PutCarrotOnSpongeLargerInScene-v2", {}), # seen source and target, unseen combination
    "widowx_eggplant_on_sponge_clean": ("PutEggplantOnSpongeLargerInScene-v2", {}),
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
    "widowx_spoon_on_towel_lang_action": ("PutSpoonOnTableClothInScene-LangV3", {}), # language action
    "widowx_stack_cube_lang_action": ("StackGreenCubeOnYellowCubeBakedTexInScene-LangV1", {}),
    "widowx_eggplant_in_basket_lang_color": ("PutEggplantInBasketScene-LangV1", {}), # eggplant -> purple object
    "widowx_eggplant_in_basket_lang_common": ("PutEggplantInBasketScene-LangV2", {}), # yellow basket -> where the dishes usually get dried
    "widowx_eggplant_in_basket_lang_action": ("PutEggplantInBasketScene-LangV3", {}),
    "widowx_carrot_on_keyboard_lang_common": ("PutCarrotOnKeyboardInScene-LangV1", {}), # keyboard -> tool for typing words
    "widowx_coke_can_on_plate_lang_common": ("PutCokeCanOnPlateInScene-LangV3", {}), # languge commonsense, no distract, (thirsty)
    "widowx_coke_can_on_plate_lang_neg": ("PutCokeCanOnPlateInScene-LangV1", {}), # "put coke can, not the carrot, not the pepsi can, on the plate"
    "widowx_coke_can_on_plate_lang_common_distract": ("PutCokeCanOnPlateInScene-LangV2", {}), # thirsty "put the object that one needs the most when they are thirsty on plate" + carrot + eggplant
    
    "widowx_orange_juice_on_plate_clean": ("PutOrangeJuiceOnPlateInScene-v2", {}), # ood source
    "widowx_orange_juice_on_plate_distract": ("PutOrangeJuiceOnPlateInScene-distract", {}), # + carrot + orange
    "widowx_orange_juice_on_plate_lang_neg": ("PutOrangeJuiceOnPlateInScene-LangV1", {}), # "put orange juice, not the orange, on the plate"
    "widowx_orange_juice_on_plate_lang_common": ("PutOrangeJuiceOnPlateInScene-LangV2", {}), # "put the juice squeezed from orange on the plate"
    "widowx_orange_juice_on_plate_lang_common_distract": ("PutOrangeJuiceOnPlateInScene-LangV3", {}), # "put the juice squeezed from orange on the plate" + carrot + orange
    "widowx_orange_juice_on_plate_lang_common_distractv2": ("PutOrangeJuiceOnPlateInScene-LangV4", {}), # "put the drink rich in vitamin C on the plate" + coke can + orange
    "widowx_nut_on_plate_clean": ("PutNutOnPlateInScene-v2", {}), # ood source
    "widowx_nut_on_plate_lang_common": ("PutNutOnPlateInScene-LangV1", {}), # ood source + language commonsense, nut -> metal component for taming bolts
    "widowx_eggplant_on_keyboard_clean": ("PutEggplantOnKeyboardInScene-v2", {}), # ood target
    "widowx_carrot_on_ramekin_clean": ("PutCarrotOnRamekinInScene-v2", {}), # ood target
    "widowx_carrot_on_wheel_clean": ("PutCarrotOnWheelInScene-v2", {}), # ood target
    "widowx_coke_can_on_ramekin_clean": ("PutCokeCanOnRamekinInScene-v2", {}), # ood source and ood target
    "widowx_coke_can_on_wheel_clean": ("PutCokeCanOnWheelInScene-v2", {}), # ood target ood source
    "widowx_nut_on_wheel_clean": ("PutNutOnWheelInScene-v2", {}), # ood target
    "widowx_cube_on_plate_lang_shape": ("PutGreenCubeOnPlateInScene-LangV1", {}), # "put the square shaped object on the round shaped object"
    "widowx_spoon_on_towel_lang_neg": ("PutSpoonOnTableClothInScene-LangV4", {}), # "put the spoon on the towel, not on the plate"
    "widowx_spoon_on_towel_lang_color": ("PutSpoonOnTableClothInScene-LangV5", {}), # "put the shiny object with green handle on the blue object"
    "widowx_carrot_on_plate_lang_color": ("PutCarrotOnPlateInScene-LangV6", {}), # "put the orange object on the yellow object"
    # # ! not-in-used tasks
    # # lighting
    # "widowx_carrot_on_plate_brighter": ("PutCarrotOnPlateInScene-light-v1", {}),
    # "widowx_carrot_on_plate_darker" : ("PutCarrotOnPlateInScene-light-v2", {}),
    # "widowx_eggplant_in_basket_brighter": ("PutEggplantInBasketScene-light-v1", {}),
    # "widowx_eggplant_in_basket_darker": ("PutEggplantInBasketScene-light-v2", {}),
    # # too difficult or difficult to evaluate
    # # "widowx_carrot_on_sponge_clean": ("PutCarrotOnSpongeInScene-v2", {}), # OLD
    # "widowx_eggplant_on_sponge": ("PutEggplantOnSpongeInScene-v2", {}), # sponge too small
    # "widowx_carrot_on_coke_can": ("PutCarrotOnCokeCanInScene-v1", {}),
    # "widowx_carrot_on_green_cube": ("PutCarrotOnGreenCubeInScene-v1", {}),
    # "widowx_plate_on_green_cube": ("PutPlateOnGreenCubeInScene-v1", {}),
    # "widowx_coke_can_on_pepsi_can": ("PutCokeCanOnPepsiCanInScene-v1", {}),
}


def make(task_name):
    """Creates simulated eval environment from task name."""
    assert task_name in ENVIRONMENTS, f"Task {task_name} is not supported. Environments: \n {ENVIRONMENTS}"
    env_name, kwargs = ENVIRONMENT_MAP[task_name]
    kwargs["prepackaged_config"] = True
    env = gym.make(env_name, obs_mode="rgbd", **kwargs)
    return env
