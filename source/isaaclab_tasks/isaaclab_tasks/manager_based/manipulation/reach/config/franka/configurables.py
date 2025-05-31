# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# Observation configurations
@configclass
class StateNoNoiseObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class StateNoisyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# action configurations

@configclass
class HydraConfigurables:
    configurations: dict[str, any] = {
        "env.observations": {
            "state_obs_no_noise": StateNoNoiseObservationsCfg(),
            "state_obs_noisy": StateNoisyObservationsCfg(),
        },
        "env.actions.arm_action": {
            "ik_abs_arm_action": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
            "ik_rel_arm_action": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
                scale=0.5,
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
            "joint_pos_arm_action": mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
            ),
            "osc_arm_action": mdp.OperationalSpaceControllerActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller_cfg=mdp.OperationalSpaceControllerCfg(
                    target_types=["pose_abs"],
                    impedance_mode="variable_kp",
                    inertial_dynamics_decoupling=True,
                    partial_inertial_dynamics_decoupling=False,
                    gravity_compensation=False,
                    motion_stiffness_task=100.0,
                    motion_damping_ratio_task=1.0,
                    motion_stiffness_limits_task=(50.0, 200.0),
                    nullspace_control="position",
                ),
                nullspace_joint_pos_target="center",
                position_scale=1.0,
                orientation_scale=1.0,
                stiffness_scale=100.0,
            )
        }
    }
