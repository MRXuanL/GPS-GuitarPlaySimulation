from typing import Sequence

import mujoco
import numpy as np
from dm_control import composer
from mujoco_utils import composer_utils, physics_utils
from guitarplay.modelpy.hands import HandSide, shadow_hand,shadow_hand_R,base_R
from guitarplay.modelpy.guitar import guitar
from guitarplay.music.goal import Goal
from mujoco_utils import mjcf_utils, physics_utils, spec_utils, types
_FRAME_RATE=30
# Timestep of the physics simulation, in seconds.
_PHYSICS_TIMESTEP =1.0/_FRAME_RATE/10

# Interval between agent actions, in seconds.
_CONTROL_TIMESTEP = 1.0/_FRAME_RATE 

# Default position and orientation of the hands.
_LEFT_HAND_POSITION = (-0.3,-0.32,0.60)
_LEFT_HAND_QUATERNION = (  7.07106781e-01,-7.07106781e-01, 4.32963729e-17,  4.32963729e-17)
_RIGHT_HAND_POSITION = (0.33096,0.08,1.01727)
_RIGHT_HAND_QUATERNION = (2.19006687e-01,  1.29458477e-06, -9.75723358e-01,  2.90576955e-07)

_ATTACHMENT_YAW = 1.57  # Degrees.
_ANIMATION_NUM=10




class GuitarOnlyTask(composer.Task):
    """Guitar task with no hands."""

    def __init__(
        self,
        change_color_on_activation: bool = True,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
        key_bound:float=0.01
    ) -> None:
        self._arena = guitar.Guitar(
            change_color_on_activation=change_color_on_activation,
            key_bound=key_bound
        )


        self._arena.mjcf_model.default.geom.solref = (physics_timestep * 2, 1)

        self.set_timesteps(
            control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    # Accessors.

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena.mjcf_model


    @property
    def guitar(self):
        return self._arena
    # Composer methods.

    def get_reward(self, physics) -> float:
        del physics  # Unused.
        return 0.0
    
class GuitarTask(GuitarOnlyTask):
    """Base class for guitar tasks."""

    def __init__(
        self,
        gravity_compensation: bool = False,
        change_color_on_activation: bool = True,
        primitive_fingertip_collisions: bool = False,
        reduced_action_space: bool = False,
        attachment_yaw: float = _ATTACHMENT_YAW,
        forearm_dofs: Sequence[str] = shadow_hand._DEFAULT_FOREARM_DOFS,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
        key_bound:float=0.01
    ) -> None:
        super().__init__(
            change_color_on_activation=change_color_on_activation,
            physics_timestep=physics_timestep,
            control_timestep=control_timestep,
            key_bound=key_bound
        )
        self._trajactory_site=[]
        for i in range(_ANIMATION_NUM):
            trajactory_site=self.arena.worldbody.add('site',name='tsite'+str(i),type='sphere',size='0.001',rgba=[0,1,0,0])
            self._trajactory_site.append(trajactory_site)
        self._right_hand = self._add_righthand(
            hand_side=base_R.HandSide.RIGHT,
            position=_RIGHT_HAND_POSITION,
            quaternion=_RIGHT_HAND_QUATERNION,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
        )
        self._left_hand = self._add_hand(
            hand_side=HandSide.LEFT,
            position=_LEFT_HAND_POSITION,
            quaternion=_LEFT_HAND_QUATERNION,
            gravity_compensation=gravity_compensation,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            reduced_action_space=reduced_action_space,
            attachment_yaw=attachment_yaw,
            forearm_dofs=forearm_dofs,
        )
        # print(self.arena.to_xml_string())
        # with open('scene.xml','w') as f:
        #     f.write(self.arena.to_xml_string())



      

    # Accessors.

    @property
    def left_hand(self) -> shadow_hand.ShadowHand:
        return self._left_hand

    @property
    def right_hand(self) -> shadow_hand.ShadowHand:
        return self._right_hand

    # Helper methods.

    def _add_hand(
        self,
        hand_side: HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> shadow_hand.ShadowHand:
        joint_range = [-0.23, 0.09]

        # Offset the joint range by the hand's initial position.
        # joint_range[0] -= position[1]
        # joint_range[1] -= position[1]

        hand = shadow_hand.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        if forearm_tx_actuator is not None:
            forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand
    
    def _add_righthand(
        self,
        hand_side: base_R.HandSide,
        position,
        quaternion,
        gravity_compensation: bool,
        primitive_fingertip_collisions: bool,
        reduced_action_space: bool,
        attachment_yaw: float,
        forearm_dofs: Sequence[str],
    ) -> shadow_hand.ShadowHand:
        joint_range = [-0.00000001, 0.00000001]

        # Offset the joint range by the hand's initial position.
        joint_range[0] -= position[1]
        joint_range[1] -= position[1]

        hand = shadow_hand_R.ShadowHand(
            side=hand_side,
            primitive_fingertip_collisions=primitive_fingertip_collisions,
            restrict_wrist_yaw_range=False,
            reduced_action_space=reduced_action_space,
            forearm_dofs=forearm_dofs,
        )
        hand.root_body.pos = position

        # Slightly rotate the forearms inwards (Z-axis) to mimic human posture.
        rotate_axis = np.asarray([0, 0, 1], dtype=np.float64)
        rotate_by = np.zeros(4, dtype=np.float64)
        sign = -1 if hand_side == HandSide.LEFT else 1
        angle = np.radians(sign * attachment_yaw)
        mujoco.mju_axisAngle2Quat(rotate_by, rotate_axis, angle)
        final_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mulQuat(final_quaternion, rotate_by, quaternion)
        hand.root_body.quat = final_quaternion

        if gravity_compensation:
            physics_utils.compensate_gravity(hand.mjcf_model)

        # Override forearm translation joint range.
        forearm_tx_joint = hand.mjcf_model.find("joint", "forearm_tx")
        if forearm_tx_joint is not None:
            forearm_tx_joint.range = joint_range
        # forearm_tx_actuator = hand.mjcf_model.find("actuator", "forearm_tx")
        # if forearm_tx_actuator is not None:
        #     forearm_tx_actuator.ctrlrange = joint_range

        self._arena.attach(hand)
        return hand

